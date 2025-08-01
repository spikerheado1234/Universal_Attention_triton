import torch
import triton
import triton.language as tl
from torch.nn.functional import scaled_dot_product_attention

import numpy as np
import inspect
import time
from math import sqrt
import pdb

def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"

def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"

def supports_host_descriptor():
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 9

def get_cuda_configs():
    configs = []

    for BLOCK_M in [32, 64, 128, 256]:
        for BLOCK_N in [32, 64, 128, 256]:
            configs.append(triton.Config({"BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N}, num_stages=2, num_warps=8))

    return configs

@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q,  #
                    desc_k, desc_v,  #
                    offset_y, dtype: tl.constexpr, start_m, qk_scale,  #
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  #
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  #
                    N_CTX: tl.constexpr, causal: tl.constexpr, KV_H: tl.constexpr, Q_H: tl.constexpr):
    # range of values handled by this stage
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    # causal = False
    else:
        lo, hi = 0, N_CTX
    offsetk_y = offset_y + lo * HEAD_DIM
    offsetv_y = offset_y + lo * HEAD_DIM
    # loop over k, v and update accumulator
    for start_n in tl.range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k_ptr = offsetk_y + tl.arange(0, BLOCK_N)[:, None]*HEAD_DIM + tl.arange(0, HEAD_DIM)[None, :]
        k = tl.trans(tl.load(desc_k + k_ptr, mask=(tl.arange(0, BLOCK_N)+start_n)[:,None] < N_CTX, other=0.0))
        qk = tl.dot(q, k)
        qk *= 1/tl.sqrt(tl.cast(HEAD_DIM, dtype=tl.float32))
        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk = qk - m_ij[:, None]

        p = tl.math.exp(qk)
        # -- compute correction factor
        alpha = tl.math.exp(m_i - m_ij)
        l_ij = tl.sum(p, 1)
        # -- update output accumulator --
        acc = acc * alpha[:, None]
        # prepare p and v for the dot
        v_ptr = offsetv_y + tl.arange(0, BLOCK_N)[:, None]*HEAD_DIM + tl.arange(0, HEAD_DIM)[None, :]
        v = tl.load(desc_v + v_ptr, mask=(tl.arange(0, BLOCK_N)+start_n)[:, None] < N_CTX, other=0.0)
        p = p.to(dtype)
        # note that this non transposed v for FP8 is only supported on Blackwell
        acc = tl.dot(p, v, acc)
        # update m_i and l_i
        # place this at the end of the loop to reduce register pressure
        l_i = l_i * alpha + l_ij
        m_i = m_ij
        offsetk_y += BLOCK_N * HEAD_DIM
        offsetv_y += BLOCK_N * HEAD_DIM
    return acc, l_i, m_i

#@triton.autotune(
#        configs=get_cuda_configs(),
#        key=["N_CTX", "HEAD_DIM", "FP8_OUTPUT", "warp_specialize"]
#        )
@triton.jit
def _attn_fwd(sm_scale, M,  #
              Z, H, desc_q, desc_k, desc_v, desc_o, N_CTX,  #
              HEAD_DIM: tl.constexpr,  #
              BLOCK_M: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
              FP8_OUTPUT: tl.constexpr,  #
              STAGE: tl.constexpr,  #
              warp_specialize: tl.constexpr,  #
              causal: tl.constexpr,
              KV_H: tl.constexpr,
              Q_H: tl.constexpr
              ):
    dtype = tl.float8e5 if FP8_OUTPUT else tl.float16
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    offsetqo_y = off_z * (N_CTX * H * HEAD_DIM) + off_h * N_CTX * HEAD_DIM
    qo_offset_y = offsetqo_y + start_m * BLOCK_M * HEAD_DIM
    ## Compute offset_y of k/v taking into account GQA. ##
    offset_y = off_z * (N_CTX * H * HEAD_DIM) + (off_h % KV_H) * N_CTX * HEAD_DIM
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)
    # load q: it will stay in SRAM throughout
    qo_ptr = qo_offset_y + tl.arange(0, BLOCK_M)[:, None] * HEAD_DIM  + tl.arange(0, HEAD_DIM)[None, :]
    q = tl.load(desc_q + qo_ptr, mask=tl.arange(0, BLOCK_M)[:, None] + start_m * BLOCK_M < N_CTX, other=0.0)
    # stage 1: off-band
    # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q,  #
                                        desc_k, desc_v,  #
                                        offset_y, dtype, start_m, qk_scale,  #
                                        BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                        4 - STAGE, offs_m, offs_n, N_CTX, causal, KV_H, Q_H)
    # stage 2: on-band
    if STAGE & 2:
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q,  #
                                        desc_k, desc_v,  #
                                        offset_y, dtype, start_m, qk_scale,  #
                                        BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                        2, offs_m, offs_n, N_CTX, causal, KV_H, Q_H)
    # epilogue
    m_i += tl.math.log(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    tl.store(desc_o+qo_ptr, acc.to(dtype), mask=tl.arange(0, BLOCK_M)[:, None]+start_m*BLOCK_M < N_CTX)

@triton.jit
def _attn_bwd_preprocess(O, DO,  #
                         Delta,  #
                         Z, H, N_CTX,  #
                         BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr  #
                         ):
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_hz = tl.program_id(1)
    off_n = tl.arange(0, HEAD_DIM)
    # load
    o = tl.load(O + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :], mask=off_m[:, None] < N_CTX, other=0.0)
    do = tl.load(DO + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :], mask=off_m[:, None] < N_CTX, other=0.0).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    # write-back
    tl.store(Delta + off_hz * N_CTX + off_m, delta, mask=off_m < N_CTX)


# The main inner-loop logic for computing dK and dV.
@triton.jit
def _attn_bwd_dkdv(dk, dv,  #
                   Q, k, v, sm_scale,  #
                   DO,  #
                   M, D,  #
                   # shared by Q/K/V/DO.
                   stride_tok, stride_d,  #
                   H, N_CTX, BLOCK_M1: tl.constexpr,  #
                   BLOCK_N1: tl.constexpr,  #
                   HEAD_DIM: tl.constexpr,  #
                   # Filled in by the wrapper.
                   start_n, start_m, num_steps, MASK):
    offs_m = start_m + tl.arange(0, BLOCK_M1)
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    offs_k = tl.arange(0, HEAD_DIM)
    qT_ptrs = Q + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    do_ptrs = DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    # BLOCK_N1 must be a multiple of BLOCK_M1, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)
    curr_m = start_m
    step_m = BLOCK_M1
    for blk_idx in range(num_steps):
        offs_m = curr_m + tl.arange(0, BLOCK_M1)
        qT = tl.trans(tl.load(qT_ptrs, mask=offs_m[:, None] < N_CTX))
        # Load m before computing qk to reduce pipeline stall.
        m = tl.load(M + offs_m, mask=offs_m < N_CTX)
        qkT = tl.dot(k, qT) / tl.sqrt(tl.cast(HEAD_DIM, tl.float32))
        pT = tl.math.exp(qkT - m[None, :])
        # Autoregressive masking.
        if MASK:
            mask = (offs_m[None, :] >= offs_n[:, None]) & (offs_m[None, :] < N_CTX) & (offs_n[:, None] < N_CTX)
            pT = tl.where(mask, pT, 0.0)
        do = tl.load(do_ptrs, mask=offs_m[:, None] < N_CTX)
        # Compute dV.
        ppT = pT
        dv += tl.dot(ppT, tl.cast(do, tl.float32))
        # D (= delta) is pre-divided by ds_scale.
        Di = tl.load(D + offs_m, mask=offs_m < N_CTX)
        # Compute dP and dS.
        dpT = tl.dot(v, tl.trans(do)).to(tl.float32)
        dsT = pT * (dpT - Di[None, :])
        #dsT = dsT.to(tl.float16)
        dk += tl.dot(dsT, tl.cast(tl.trans(qT), tl.float32))
        # Increment pointers.
        curr_m += step_m
        qT_ptrs += step_m * stride_tok
        do_ptrs += step_m * stride_tok
    return dk, dv


# the main inner-loop logic for computing dQ
@triton.jit
def _attn_bwd_dq(dq, q, K, V,  #
                 do, m, D,
                 # shared by Q/K/V/DO.
                 stride_tok, stride_d,  #
                 H, N_CTX,  #
                 BLOCK_M2: tl.constexpr,  #
                 BLOCK_N2: tl.constexpr,  #
                 HEAD_DIM: tl.constexpr,
                 # Filled in by the wrapper.
                 start_m, start_n, num_steps,  #
                 MASK: tl.constexpr):
    offs_m = start_m + tl.arange(0, BLOCK_M2)
    offs_n = start_n + tl.arange(0, BLOCK_N2)
    offs_k = tl.arange(0, HEAD_DIM)
    kT_ptrs = K + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    vT_ptrs = V + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    # D (= delta) is pre-divided by ds_scale.
    Di = tl.load(D + offs_m, mask=offs_m < N_CTX, other=0.0)
    # BLOCK_M2 must be a multiple of BLOCK_N2, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_M2 % BLOCK_N2 == 0)
    curr_n = start_n
    step_n = BLOCK_N2
    for blk_idx in range(num_steps):
        offs_n = curr_n + tl.arange(0, BLOCK_N2)
        kT = tl.load(kT_ptrs, mask=offs_n[None, :] < N_CTX)
        vT = tl.load(vT_ptrs, mask=offs_n[None, :] < N_CTX)
        qk = tl.dot(q, kT) / tl.sqrt(tl.cast(HEAD_DIM, tl.float32))
        p = tl.math.exp(qk - m)
        # Autoregressive masking.
        if MASK:
            offs_n = curr_n + tl.arange(0, BLOCK_N2)
            mask = (offs_m[:, None] >= offs_n[None, :]) & (offs_m[:, None] < N_CTX) & (offs_n[None, :] < N_CTX)
            p = tl.where(mask, p, 0.0)
        # Compute dP and dS.
        dp = tl.dot(do, vT).to(tl.float32)
        ds = p * (dp - Di[:, None])
        ds = ds.to(tl.float16)
        # Compute dQ.
        # NOTE: We need to de-scale dq in the end, because kT was pre-scaled.
        dq += tl.dot(ds, tl.trans(kT))
        # Increment pointers.
        curr_n += step_n
        kT_ptrs += step_n * stride_tok
        vT_ptrs += step_n * stride_tok
    return dq

@triton.jit
def _attn_bwd(Q, K, V, sm_scale,  #
              DO,  #
              DQ, DK, DV,  #
              M, D,
              # shared by Q/K/V/DO.
              stride_z, stride_h, stride_tok, stride_d,  #
              H, N_CTX,  #
              BLOCK_M1: tl.constexpr,  #
              BLOCK_N1: tl.constexpr,  #
              BLOCK_M2: tl.constexpr,  #
              BLOCK_N2: tl.constexpr,  #
              BLK_SLICE_FACTOR: tl.constexpr,  #
              HEAD_DIM: tl.constexpr,
              causal: tl.constexpr):

    bhid = tl.program_id(2)
    off_chz = (bhid * N_CTX).to(tl.int64)
    adj = (stride_h * (bhid % H) + stride_z * (bhid // H)).to(tl.int64)
    pid = tl.program_id(0)

    # offset pointers for batch/head
    Q += adj
    K += adj
    V += adj
    DO += adj
    DQ += adj
    DK += adj
    DV += adj
    M += off_chz
    D += off_chz

    # load scales
    offs_k = tl.arange(0, HEAD_DIM)

    start_n = pid * BLOCK_N1
    start_m = start_n

    offs_n = start_n + tl.arange(0, BLOCK_N1)

    dv = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)

    # load K and V: they stay in SRAM throughout the inner loop.
    k = tl.load(K + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d, mask=offs_n[:, None] < N_CTX)
    v = tl.load(V + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d, mask=offs_n[:, None] < N_CTX)

    dk, dv = _attn_bwd_dkdv(dk, dv,  #
                        Q, k, v, sm_scale,  #
                        DO,  #
                        M, D,  #
                        stride_tok, stride_d,  #
                        H, N_CTX,  #
                        BLOCK_M1, BLOCK_N1, HEAD_DIM,  #
                        start_n, start_m, num_steps=1,  #
                        MASK=True  #
                        )

    start_m = start_n + BLOCK_M1 
    num_steps = tl.cdiv((N_CTX - start_m), BLOCK_M1)

    # Compute dK and dV for non-masked blocks.
    dk, dv = _attn_bwd_dkdv(  #
        dk, dv,  #
        Q, k, v, sm_scale,  #
        DO,  #
        M, D,  #
        stride_tok, stride_d,  #
        H, N_CTX,  #
        BLOCK_M1, BLOCK_N1, HEAD_DIM,  #
        start_n, start_m, num_steps,  #
        MASK=True  #
    )

    dv_ptrs = DV + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.store(dv_ptrs, dv.to(tl.float16), mask=offs_n[:, None] < N_CTX)

    # Write back dK.
    dk *= sm_scale
    dk_ptrs = DK + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.store(dk_ptrs, dk.to(tl.float16), mask=offs_n[:, None] < N_CTX)

    # THIS BLOCK DOES DQ:
    start_m = pid * BLOCK_M2
    offs_m = start_m + tl.arange(0, BLOCK_M2)

    # Load data for this query block
    q = tl.load(Q + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d, mask=offs_m[:, None] < N_CTX)
    dq = tl.zeros([BLOCK_M2, HEAD_DIM], dtype=tl.float32)
    do = tl.load(DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d, mask=offs_m[:, None] < N_CTX)
    m = tl.load(M + offs_m, mask=offs_m < N_CTX)[:, None]

    # For causal attention, the q-block iterates backward over keys from the diagonal.
    # The highest key index this query block can see is limited by N_CTX.
    effective_end_n = tl.minimum(start_m + BLOCK_M2, N_CTX)
    
    if causal:
        # We start from key 0 and go up to the diagonal.
        num_steps = tl.cdiv(effective_end_n, BLOCK_N2)
        dq = _attn_bwd_dq(
            dq, q, K, V, do, m, D,
            stride_tok, stride_d, H, N_CTX, BLOCK_M2, BLOCK_N2, HEAD_DIM,
            start_m, 0, num_steps, MASK=True
        )
    else: # Non-causal
        num_steps = tl.cdiv(N_CTX, BLOCK_N2)
        dq = _attn_bwd_dq(
            dq, q, K, V, do, m, D,
            stride_tok, stride_d, H, N_CTX, BLOCK_M2, BLOCK_N2, HEAD_DIM,
            start_m, 0, num_steps, MASK=False
        )

    # Write back dQ.
    dq_ptrs = DQ + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    dq *= sm_scale
    tl.store(dq_ptrs, dq, mask=offs_m[:, None] < N_CTX)

class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale, warp_specialize=True):
        assert causal, 'currently, only support causal-autoregressive style generation only.'
        assert q.shape[2] > 0 and (q.shape[2] & (q.shape[2] - 1)) == 0, 'Currently only works for powers of 2. Trying to debug other paths. Masking is non-trivial'
        # shape constraints
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        # when v is in float8_e5m2 it is transposed.
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}
        o = torch.empty_like(q)
        stage = 3 if causal else 1
        extra_kern_args = {}
        # Tuning for AMD target
        KV_H, Q_H = k.shape[1], q.shape[1]

        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32) ## (batch, head, sequence).
        desc_q = q
        desc_v = v
        desc_k = k
        desc_o = o

        #def grid(META):
        #    return (triton.cdiv(q.shape[2], META["BLOCK_M"]), q.shape[0] * q.shape[1], 1)

        BLOCK_M=16
        BLOCK_N=16
        grid = (triton.cdiv(q.shape[2], BLOCK_M), q.shape[0] * q.shape[1], 1)

        ctx.grid = grid

        _attn_fwd[grid](
            sm_scale, M,  #
            q.shape[0], q.shape[1],  #
            desc_q, desc_k, desc_v, desc_o,  #
            N_CTX=q.shape[2],  #
            HEAD_DIM=HEAD_DIM_K,  #
            BLOCK_M=BLOCK_M, # Comment out after debugging finishes.
            BLOCK_N=BLOCK_N, # Comment out after debugging finishes.
            FP8_OUTPUT=q.dtype == torch.float8_e5m2,  #
            STAGE=stage,  #
            warp_specialize=warp_specialize,  #
            causal=causal,
            KV_H=KV_H,
            Q_H=Q_H,
            **extra_kern_args)

        ctx.save_for_backward(q, k, v, o, M)
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.causal = causal
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, M = ctx.saved_tensors
        assert do.is_contiguous()
        assert q.stride() == k.stride() == v.stride() == o.stride() == do.stride()
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        BATCH, N_HEAD, N_CTX = q.shape[:3]
        PRE_BLOCK = 128
        NUM_WARPS, NUM_STAGES = 4, 5
        BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 32, 128, 128, 32
        BLK_SLICE_FACTOR = 2
        RCP_LN2 = 1.4426950408889634  # = 1.0 / ln(2)
        PRE_BLOCK = 128
        ## Remove this strict assertion.
        #assert N_CTX % PRE_BLOCK == 0
        pre_grid = (triton.cdiv(N_CTX, PRE_BLOCK), BATCH * N_HEAD)
        delta = torch.empty_like(M)
        _attn_bwd_preprocess[pre_grid](
            o, do,  #
            delta,  #
            BATCH, N_HEAD, N_CTX,  #
            BLOCK_M=PRE_BLOCK, HEAD_DIM=ctx.HEAD_DIM  #
        )
        grid = (triton.cdiv(N_CTX, BLOCK_N1), 1, BATCH * N_HEAD)
        scale = 1.0 / ctx.HEAD_DIM**0.5
        arg_k = k
        _attn_bwd[grid](
            q, arg_k, v, scale, do, dq, dk, dv,  #
            M, delta,  #
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
            N_HEAD, N_CTX,  #
            BLOCK_M1=BLOCK_M1, BLOCK_N1=BLOCK_N1,  #
            BLOCK_M2=BLOCK_M2, BLOCK_N2=BLOCK_N2,  #
            BLK_SLICE_FACTOR=BLK_SLICE_FACTOR,  #
            HEAD_DIM=ctx.HEAD_DIM,  #
            causal=ctx.causal,
            num_warps=NUM_WARPS,  #
            num_stages=NUM_STAGES  #
        )

        return dq, dk, dv, None, None, None, None
    
attention = _attention.apply


## We run a simple correctness test here. ##
if __name__ == '__main__':

    dtype = torch.float16
    mode="bwd"
    BATCH=2
    H=2
    N_CTX=16
    HEAD_DIM=32
    device="cuda" if torch.cuda.is_available() else "cpu"
    provider = "triton" ## triton/flash.
    q = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    k = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    v = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    q_torch = q.clone().detach().requires_grad_(True)
    k_torch = k.clone().detach().requires_grad_(True)
    v_torch = v.clone().detach().requires_grad_(True)
    sm_scale = 1.3
    causal = True
    fn = lambda: attention(q, k, v, causal, sm_scale)
    if mode == "bwd":
        o = fn()
        do = torch.randn_like(o)
        fn = lambda: o.backward(do, retain_graph=True)

    triton_output = fn()

    fn = lambda: scaled_dot_product_attention(q_torch,k_torch,v_torch, is_causal=causal)

    if mode == "bwd":
        o = fn()
        do = torch.randn_like(o)
        fn = lambda: o.backward(do, retain_graph=True)

    true_output = fn()

    if mode != "bwd":
        print(f'delta is: {torch.allclose(triton_output, true_output, rtol=1e-1, atol=1e-1)}')

    if mode == "bwd":
        print(f'q_torch.grad: {q_torch.grad}')
        print(f'q.grad: {q.grad}')
        print(f'dq delta is: {torch.allclose(q_torch.grad, q.grad, rtol=1, atol=1)}')
        print(f'dk delta is: {torch.allclose(k_torch.grad, k.grad, rtol=1, atol=1)}')
        print(f'dv delta is: {torch.allclose(v_torch.grad, v.grad, rtol=1, atol=1)}')

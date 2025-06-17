import torch
import triton
import triton.language as tl

@triton.jit
def _affinity_kernel(
    k_ptr, kt_ptr, src_ptr, dest_ptr,
    aff1_ptr, aff2_ptr, aff3_ptr, score_ptr,
    B, H, R, C, D, L, I_C,
    stride_b_k, stride_h_k, stride_r_k, stride_c_k, stride_d_k,
    stride_b_kt, stride_h_kt, stride_d_kt, stride_l_kt,
    stride_b_src, stride_h_src, stride_r_src, stride_c_src,
    stride_b_dest, stride_h_dest, stride_r_dest, stride_l_dest,
    stride_b_aff, stride_h_aff, stride_r_aff, stride_c_aff, stride_l_aff,
    stride_b_score, stride_h_score, stride_r_score, stride_c_score, stride_l_score,
    BLOCK_L: tl.constexpr, BLOCK_N: tl.constexpr, STORE_INTERMEDIATE: tl.constexpr
):
    # Program IDs
    b = tl.program_id(0)
    h = tl.program_id(1)
    r = tl.program_id(2)

    # Offsets within the block
    offs_c = tl.arange(0, BLOCK_L)
    offs_l = tl.arange(0, BLOCK_N)

    # Load a tile of k: shape [BLOCK_L, D]
    k_off = k_ptr + b * stride_b_k + h * stride_h_k + r * stride_r_k
    k_tile = tl.load(
        k_off + offs_c[:, None] * stride_c_k + tl.arange(0, D)[None, :] * stride_d_k,
        mask=(offs_c[:, None] < C),
        other=0.0
    )

    # Load a tile of kt: shape [D, BLOCK_N]
    kt_off = kt_ptr + b * stride_b_kt + h * stride_h_kt
    kt_tile = tl.load(
        kt_off + tl.arange(0, D)[:, None] * stride_d_kt + offs_l[None, :] * stride_l_kt,
        mask=(offs_l[None, :] < L),
        other=0.0
    )

    # aff1 = k_tile @ kt_tile -> [BLOCK_L, BLOCK_N]
    aff1_tile = tl.dot(k_tile, kt_tile)

    # aff2 = relu(aff1)^(2/3)
    aff2_tile = tl.pow(tl.maximum(aff1_tile, 0.0), 2.0/3.0)

    # Load static_src and static_dest
    src_off = src_ptr + b * stride_b_src + h * stride_h_src + r * stride_r_src
    src_tile = tl.load(
        src_off + offs_c * stride_c_src,
        mask=offs_c < C,
        other=0.0
    )  # [BLOCK_L]

    dest_off = dest_ptr + b * stride_b_dest + h * stride_h_dest + r * stride_r_dest
    dest_tile = tl.load(
        dest_off + offs_l * stride_l_dest,
        mask=offs_l < L,
        other=0.0
    )  # [BLOCK_N]

    # aff3 = aff2 * src * dest
    aff3_tile = aff2_tile * src_tile[:, None] * dest_tile[None, :]

    # clamp and log1p(-x)
    clamped = tl.minimum(tl.maximum(aff3_tile, 0.0), 1.0 - 1e-6)
    score_tile = tl.log1p(-clamped)

    # Apply triu: keep elements with n >= c + I_C
    mask_triu = offs_l[None, :] >= (offs_c[:, None] + I_C)
    score_tile = tl.where(mask_triu, score_tile, tl.full(score_tile.shape, -1e12))

    # Prefix-sum along the N dimension (naive loop)
    # Note: BLOCK_N should be small enough for unrolling
    for n_idx in range(1, BLOCK_N):
        score_tile[:, n_idx] += score_tile[:, n_idx - 1]

    # Apply tril mask: set score to -1e12 where n <= c + I_C - 1
    mask_tril = offs_l[None, :] <= (offs_c[:, None] + I_C - 1)
    score_tile = tl.where(mask_tril, tl.full(score_tile.shape, -1e12), score_tile)

    if STORE_INTERMEDIATE:
        tl.store(
            aff1_ptr + b * stride_b_aff + h * stride_h_aff + r * stride_r_aff + 
            offs_c[:, None] * stride_c_aff + offs_l[None, :] * stride_l_aff,
            aff1_tile
        )
        tl.store(
            aff2_ptr + b * stride_b_aff + h * stride_h_aff + r * stride_r_aff + 
            offs_c[:, None] * stride_c_aff + offs_l[None, :] * stride_l_aff,
            aff2_tile
        )
        tl.store(
            aff3_ptr + b * stride_b_aff + h * stride_h_aff + r * stride_r_aff + 
            offs_c[:, None] * stride_c_aff + offs_l[None, :] * stride_l_aff,
            aff3_tile
        )

    tl.store(
        score_ptr + b * stride_b_score + h * stride_h_score + r * stride_r_score + 
        offs_c[:, None] * stride_c_score + offs_l[None, :] * stride_l_score,
        score_tile
    )


# Python wrapper

def affinity_kernel_fwd(k, kt, static_src, static_dest, i, c):
    B, H, R, C, D = k.shape
    _, _, _, L = kt.shape
    I_C = i * c + 1

    default_block_l = 128
    default_block_n = 128

    # aff1 = torch.empty((B, H, R, C, L), device=k.device, dtype=k.dtype)
    # aff2 = torch.empty((B, H, R, C, L), device=k.device, dtype=k.dtype)
    # aff3 = torch.empty((B, H, R, C, L), device=k.device, dtype=k.dtype)

    score = torch.empty((B, H, R, C, L), device=k.device, dtype=k.dtype)

    # Launch Triton kernel
    grid = (B, H, R)
    _affinity_kernel[grid](
        k, kt, static_src, static_dest,
        None, None, None, score,
        B, H, R, C, D, L, I_C,
        *k.stride(),
        *kt.stride(),
        *static_src.stride(),
        *static_dest.stride(),
        *out.stride(),
        *score.stride(),
        default_block_l, default_block_n, False
    )

    return aff1, aff2, aff3, score

@triton.jit
def _affinity_kernel(
    k_, kt, static_src_, static_dest, i, c
):
    aff1 = k_.matmul(kt)
    aff2 = aff1.relu().pow(2/3).float()
    aff3 = aff2 * static_src_.unsqueeze(-1) * static_dest.unsqueeze(-2)
    score = torch.log1p(aff3.clamp(min=0,max=1-1e-6).neg()).triu(i*c+1).cumsum(3).masked_fill(mask.tril(i*c-1), -1e12)
    return aff1, aff2, aff3, score

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=2),
    ],
    key=['hdim', 'dstate', 'chunk_size'],
)
@triton.jit
def _universal_attention_fwd_kernel(
    # Pointers to matrices
    kc_ptr, vc_ptr, xq_ptr, static_src_ptr, static_dest_ptr, out_ptr, denom_ptr,
    # Matrix dimensions
    b, n_kv, rep, s, d, n_c, c,
    # Strides
    stride_kc_b, stride_kc_n_kv, stride_kc_n_c, stride_kc_c, stride_kc_d,
    stride_vc_b, stride_vc_n_kv, stride_vc_n_c, stride_vc_c, stride_vc_d,
    stride_xq_b, stride_xq_n_kv, stride_xq_rep, stride_xq_s, stride_xq_d,
    stride_static_src_b, stride_static_src_n_kv, stride_static_src_n_c, stride_static_src_c,
    stride_static_dest_b, stride_static_dest_n_kv, stride_static_dest_s,
    stride_out_b, stride_out_n_kv, stride_out_rep, stride_out_s, stride_out_d, stride_out_n_c,
    stride_denom_b, stride_denom_n_kv, stride_denom_rep, stride_denom_s, stride_denom_n_c,
    # Meta-parameters

):
    b,h,r,l,d = xq.shape
    _,_,n,c,_ = kc.shape
    mask = torch.ones(c,l, dtype=torch.bool, device=xq.device)
    out = torch.empty(b,h,r,l,d,n, dtype=xq.dtype, device=xq.device)
    denom = torch.empty(b,h,r,l,n, dtype=xq.dtype, device=xq.device)
    static_src = static_src.pow(1/3)
    static_dest = static_dest.pow(1/3)
    kt = kc.view(b,h,l,d).transpose(-2,-1)  # b h d l
    for i in range(n):
        k_ = kc[:,:,i]  # b h c d
        v_ = vc[:,:,i]  # b h c d
        static_src_ = static_src[:,:,i]  # b h c

        # Calculate decay matrix
        affinity = k_.matmul(kt).relu().pow(2/3).float()  # deltanet style decay
        affinity = affinity * static_src_.unsqueeze(-1) * static_dest.unsqueeze(-2)  # incorporate mamba-style and per-token decay
        affinity = torch.log1p(affinity.clamp(min=0, max=1-1e-6).neg())  # b h c l
        affinity = affinity.triu(i*c+1).cumsum(3)  # Accumulate decay with causal masking
        affinity = affinity.masked_fill(mask.tril(i*c-1), -1e12)  # Re-mask, with 1s on diagonal

        # Perform actual attention operation
        score = k_.unsqueeze(2).matmul(xq.transpose(-1,-2)).add(affinity.unsqueeze(2))  # b h r c l
        denom_ = score.logsumexp(dim=-2)  # b h r l
        out_ = score.transpose(-1,-2).softmax(dim=-1).to(dtype=xq.dtype).matmul(v_.unsqueeze(2))  # b h r l d

        out[...,i] = out_
        denom[...,i] = denom_
    return out, denom

def _universal_attention_fwd(kc, vc, xq, static_src, static_dest):
    b, n_kv, rep, s, d = xq.shape
    _, _, n_c, c, _ = kc.shape
    device = xq.device

    out = torch.empty(b, n_kv, rep, s, d, n_c, dtype=xq.dtype, device=device)
    denom = torch.empty(b, n_kv, rep, s, n_c, dtype=xq.dtype, device=device)

    grid = lambda META: (
    grid = lambda META: (triton.cdiv(headdim, META['BLOCK_SIZE_M']) * triton.cdiv(dstate, META['BLOCK_SIZE_N']),
                    batch * nchunks, nheads)
    
    with torch.cuda.device(device.index):
        _universal_attention_fwd_kernel[grid](
            x, B, states, dt, dA_cumsum, seq_idx,
            headdim, dstate, chunk_size,
            batch, seqlen, nheads // ngroups,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            B.stride(0), B.stride(1), B.stride(2), B.stride(-1),
            states.stride(0), states.stride(1), states.stride(2), states.stride(3), states.stride(4),
            dt.stride(0), dt.stride(2), dt.stride(1), dt.stride(3),
            dA_cumsum.stride(0), dA_cumsum.stride(2), dA_cumsum.stride(1), dA_cumsum.stride(3),
            *((seq_idx.stride(0), seq_idx.stride(1)) if seq_idx is not None else (0, 0)),
            HAS_SEQ_IDX=seq_idx is not None,
        )

    return out, denom

def matmul(a, b, activation=""):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    matmul_kernel[grid](
        a, b, c,  #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
        ACTIVATION=activation  #
    )
    return c

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=2),
    ],
    key=['hdim', 'dstate', 'chunk_size'],
)
@triton.jit
def _universal_attention_bwd_kernel(
    # Pointers to matrices
    # Matrix dimensions
    # Strides
    # Meta-parameters
):
    pass

def _universal_attention_bwd(kc, vc, xq, static_src, static_dest, dout, ddenom):

    dkc, dvc, dxq, dstatic_src, dstatic_dest = [torch.zeros_like(x) for x in [kc,vc,xq,static_src,static_dest]]
    # b,h,r,l,d = xq.shape
    # _,_,n,c,_ = kc.shape
    b, n_kv, rep, s, d = xq.shape
    _, _, n_c, c, _ = kc.shape
    mask = torch.ones(c,l, dtype=torch.bool, device=xq.device)
    static_src = static_src.pow(1/3)
    static_dest = static_dest.pow(1/3)
    kt = kc.view(b,h,l,d).transpose(-2,-1)  # b h d l

    for i in range(n):
        k_ = kc[:,:,i]  # b h c d
        v_ = vc[:,:,i]  # b h c d
        static_src_ = static_src[:,:,i]  # b h c
        dout_ = dout[...,i]
        ddenom_ = ddenom[...,i]

        # Rerun forward pass
        aff1 = k_.matmul(kt)
        aff2 = aff1.relu().pow(2/3).float()
        aff3 = aff2 * static_src_.unsqueeze(-1) * static_dest.unsqueeze(-2)
        score = torch.log1p(aff3.clamp(min=0,max=1-1e-6).neg()).triu(i*c+1).cumsum(3).masked_fill(mask.tril(i*c-1), -1e12)
        score = k_.unsqueeze(2).matmul(xq.transpose(-1,-2)).add(score.unsqueeze(2))  # b h r c l
        sscore = score.softmax(dim=-2)

        # Backward pass
        dvc[:,:,i] += sscore.to(dtype=dvc.dtype).matmul(dout_).sum(2)  # bhrcl,bhrld -> bhcd
        
        dscore = v_.unsqueeze(2).matmul(dout_.transpose(-1,-2))  # bhcd,bhrld -> bhrcl   <-- from out
        dscore = dscore.sub(dscore.mul(sscore).sum(-2,True)).mul(sscore)  # <-- from softmax
        dscore += score.softmax(-2) * ddenom_.unsqueeze(-2)  # b h r c l   <-- from denom

        dxq += dscore.to(dtype=dxq.dtype).transpose(-1,-2).matmul(k_.unsqueeze(2))  # bhrcl, bhcd -> bhrld
        dkc[:,:,i] += dscore.to(dtype=dkc.dtype).transpose(2,3).flatten(3,4).matmul(xq.flatten(2,3))  # bhrcl, bhrld -> bhcd

        daff = dscore.sum(2)  # b h c l
        daff = daff.flip([3]).cumsum(3).flip([3]).triu(i*c+1)  # <-- from cumsum
        daff /= aff3.clamp(min=1e-6, max=1-1e-6)-1  # <-- from ln(1-x)
        daff *= aff3.ge(0)
        daff *= aff3.le(1-1e-6)
        dstat = daff.mul(aff2).to(dtype=static_src.dtype)  # b h c l

        dstatic_src[:,:,i] += dstat.mul(static_dest.unsqueeze(-2)).sum(-1).div(static_src_.pow(2).mul(3))  # bhcl, bhl -> bhc
        dstatic_dest += dstat.mul(static_src_.unsqueeze(-1)).sum(-2).div(static_dest.pow(2).mul(3))  # bhcl, bhc -> bhl

        daff = daff.mul(static_src_.unsqueeze(-1)*static_dest.unsqueeze(-2))  # <-- from prod with statics
        daff = daff.to(dtype=xq.dtype) * aff1.abs().add(1e-9).pow(-1/3).mul(2/3).mul(aff1.gt(0))  # <-- from relu + pow

        dkc += daff.transpose(-1,-2).matmul(k_).view(b,h,n,c,d)  # bhcl, bhcd -> bhld, <-- grad via kt
        dkc[:,:,i] += daff.matmul(kt.transpose(-1,-2))  # bhcl, bhdl -> bhcd, <-- grad via k_

    return dkc, dvc, dxq, dstatic_src, dstatic_dest
    

import torch
import triton
import triton.language as tl
import numpy as np
import inspect

# configs = [
#     triton.Config({}, num_stages=stages, num_warps=warps) \
#     for stages in [2, 3, 4, 5]\
#     for warps in [2, 4, 8, 16]\
# ]

'''
######################################
#     Forward Kernel & Interface     #
######################################
'''
# @triton.autotune(
#     configs=configs,
#     key=['s', 'd'],
# )
@triton.jit
def _universal_attention_fwd_kernel(
    # Pointers to matrices
    q_ptr, k_ptr, v_ptr, src_ptr, dest_ptr, output_ptr, 
    spinlock_ptr, semaphore_ptr, sense_rev_ptr, 
    cum_cache_ptr, max_cache_ptr, sum_cache_ptr, 
    debug_ptr,
    # Matrix dimensions
    b, n_kv, rep, s, d, 
    # Strides
    stride_q_b, stride_q_n_rep, stride_q_s, stride_q_d, 
    stride_k_b, stride_k_n_kv, stride_k_s, stride_k_d, 
    stride_v_b, stride_v_n_kv, stride_v_s, stride_v_d,
    stride_src_b, stride_src_n_kv, stride_src_s, 
    stride_dest_b, stride_dest_n_kv, stride_dest_s, 
    stride_output_b, stride_output_n_rep, stride_output_s, stride_output_d, 
    stride_spl_b, stride_spl_n_kv, stride_spl_s,
    stride_sem_b, stride_sem_n_kv, stride_sem_s,
    stride_rev_b, stride_rev_n_kv, stride_rev_s,
    stride_cum_b, stride_cum_n_kv, stride_cum_s,
    stride_max_b, stride_max_n_kv, stride_max_s, stride_max_c, 
    stride_sum_b, stride_sum_n_kv, stride_sum_s, stride_sum_c, 
    stride_debug_b, stride_debug_n_kv, stride_debug_s, stride_debug_s1,
    # Meta-parameters
    BLOCK_C: tl.constexpr, BLOCK_D: tl.constexpr, 
    C_BLOCK: tl.constexpr, D_BLOCK: tl.constexpr,
    DTYPE: tl.constexpr,
):
    pid_b_n_kv = tl.program_id(0)
    pid_b = pid_b_n_kv // n_kv
    pid_n_kv = pid_b_n_kv % n_kv
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    offs_m = pid_m * BLOCK_C + tl.arange(0, BLOCK_C)
    offs_n = pid_n * BLOCK_C + tl.arange(0, BLOCK_C)
    block_idx = tl.arange(0, C_BLOCK)

    affinity = tl.zeros((BLOCK_C, BLOCK_C), dtype=tl.float32)

    # k @ k^T
    k_ptr += pid_b * stride_k_b + pid_n_kv * stride_k_n_kv

    for d_offset in range(0, d, BLOCK_D):
        offs_d = d_offset + tl.arange(0, BLOCK_D)
        k_mat = tl.load(
            k_ptr + offs_m[:, None] * stride_k_s + offs_d[None, :] * stride_k_d, 
            mask=(offs_m[:, None] < s) & (offs_d[None, :] < d), 
            other=0.0
        )
        k_mat = tl.cast(k_mat, tl.float32)

        kt_mat = tl.load(
            k_ptr + offs_n[:, None] * stride_k_s + offs_d[None, :] * stride_k_d, 
            mask=(offs_n[:, None] < s) & (offs_d[None, :] < d), 
            other=0.0
        )
        kt_mat = tl.cast(kt_mat, tl.float32)

        # Use ieee to use fp32, otherwise the default would be tf32 even after tl.cast
        affinity += tl.dot(k_mat, tl.trans(kt_mat), input_precision="ieee")
    
    # .relu()
    affinity = tl.maximum(affinity, 0.0)

    # .pow(2/3)
    affinity = tl.exp2(tl.log2(affinity) * 2.0 / 3.0)

    # * static_src_.pow(1/3).unsqueeze(-1) * static_dest.pow(1/3).unsqueeze(-2)
    src_ptr += pid_b * stride_src_b + pid_n_kv * stride_src_n_kv
    dest_ptr += pid_b * stride_dest_b + pid_n_kv * stride_dest_n_kv

    src_mat  = tl.load(
        src_ptr + offs_m * stride_src_s, 
        mask=offs_m < s, 
        other=0.0
    )
    src_mat = tl.cast(src_mat, tl.float32)
    src_mat = tl.exp2(tl.log2(src_mat) / 3.0)

    dest_mat = tl.load(
        dest_ptr + offs_n * stride_dest_s, 
        mask=offs_n < s, 
        other=0.0
    )
    dest_mat = tl.cast(dest_mat, tl.float32)
    dest_mat = tl.exp2(tl.log2(dest_mat) / 3.0)

    affinity = affinity * src_mat[:, None] * dest_mat[None, :]

    # torch.log1p(affinity.clamp(min=0, max=1-1e-6).neg())
    affinity = tl.clamp(affinity, 0.0, 1.0 - 1e-6)
    affinity = tl.log(1.0 - affinity) 

    # .triu(1)
    # triu_mask = (offs_m[:, None] < offs_n[None, :])
    affinity = tl.where((offs_m[:, None] < offs_n[None, :]), affinity, 0.0)

    # .cumsum(3)
    spinlock_ptr += pid_b * stride_spl_b + pid_n_kv * stride_spl_n_kv + pid_m * stride_spl_s
    cum_cache_ptr += pid_b * stride_cum_b + pid_n_kv * stride_cum_n_kv 
    
    curr_sum = tl.sum(affinity, axis=1, keep_dims=False) # put the sum into the cache
    affinity = tl.cumsum(affinity, axis=1) # local cumsum

    # Make sure the sum is computed sequentially
    while tl.atomic_add(spinlock_ptr, 0, sem="acquire") < pid_n:
        pass

    prev_sum = tl.zeros((BLOCK_C,), dtype=tl.float32)
    if pid_n > 0:
        prev_sum = tl.load(
            cum_cache_ptr + offs_m * stride_cum_s, 
            mask=offs_m < s, 
            other=0.0
        ) 
    tl.store(
        cum_cache_ptr + offs_m * stride_cum_s, 
        curr_sum + prev_sum, 
        mask=offs_m < s
    )

    tl.atomic_add(spinlock_ptr, 1, sem="release")

    affinity = affinity + prev_sum[:, None]

    # .masked_fill(mask.tril(-1), -1e12)
    # tril_mask = (offs_m[:, None] > offs_n[None, :])
    affinity = tl.where((offs_m[:, None] > offs_n[None, :]), -1e12, affinity)

    # Affinity matrix completed here
    # debug_ptr += pid_b * stride_debug_b + pid_n_kv * stride_debug_n_kv
    # tl.store(
    #     debug_ptr + offs_m[:, None] * stride_debug_s + offs_n[None, :] * stride_debug_s1,
    #     affinity,
    #     mask=(offs_m[:, None] < s) & (offs_n[None, :] < s),
    # )
    # k_.unsqueeze(2).matmul(xq.transpose(-1,-2)).add(affinity.unsqueeze(2))
    q_ptr += pid_b * stride_q_b + pid_n_kv * rep * stride_q_n_rep
    v_ptr += pid_b * stride_v_b + pid_n_kv * stride_v_n_kv
    output_ptr += pid_b * stride_output_b + pid_n_kv * rep * stride_output_n_rep

    semaphore_ptr += pid_b * stride_sem_b + pid_n_kv * stride_sem_n_kv + pid_m * stride_sem_s
    sense_rev_ptr += pid_b * stride_rev_b + pid_n_kv * stride_rev_n_kv + pid_m * stride_rev_s
    max_cache_ptr += pid_b * stride_max_b + pid_n_kv * stride_max_n_kv
    sum_cache_ptr += pid_b * stride_sum_b + pid_n_kv * stride_sum_n_kv
    
    # @Haochen: storing a 3D tensor after applying .dot() to 3D tensors would fail LLVM compilation
    # The problem is fixed in triton 3.2.0, and the alternative code is listed in matmul.py
    for r in range(0, rep):
        # k @ q^T
        output_rep_ptr = output_ptr + r * stride_output_n_rep
        kq = tl.zeros((BLOCK_C, BLOCK_C), dtype=tl.float32)

        for d_offset in range(0, d, BLOCK_D):
            offs_d = d_offset + tl.arange(0, BLOCK_D)

            k_mat = tl.load(
                k_ptr + offs_m[:, None] * stride_k_s + offs_d[None, :] * stride_k_d, 
                mask=(offs_m[:, None] < s) & (offs_d[None, :] < d), 
                other=0.0,
            )
            k_mat = tl.cast(k_mat, tl.float32)

            q_mat = tl.load(
                q_ptr + r * stride_q_n_rep + offs_n[:, None] * stride_q_s + offs_d[None, :] * stride_q_d, 
                mask=(offs_n[:, None] < s) & (offs_d[None, :] < d), 
                other=0.0,
            )
            q_mat = tl.cast(q_mat, tl.float32)

            # Use ieee to use fp32, otherwise the default would be tf32
            kq += tl.dot(k_mat, tl.trans(q_mat), input_precision="ieee")
        
        kq += affinity
        qk = tl.trans(kq)

        # Softmax(QK + mask)
        localmax = tl.max(qk, axis=1)
        tl.store(
            max_cache_ptr + offs_m * stride_max_s + pid_n * stride_max_c, 
            localmax, 
            mask=(offs_m < s)
        )

        qk = tl.exp(qk - localmax[:, None])
        qk_sumexp = tl.sum(qk, axis=1)
        tl.store(
            sum_cache_ptr + offs_m * stride_sum_s + pid_n * stride_sum_c, 
            qk_sumexp, 
            mask=(offs_m < s)
        )

        if pid_n == 0:
            tl.atomic_xchg(semaphore_ptr, 0)
        while tl.load(semaphore_ptr, mask=True, other=0) > 0:
            pass
        tl.atomic_add(semaphore_ptr, 1)
        # Don't use atomic read here, or it will prevent other atomic operations
        while tl.load(semaphore_ptr, mask=True, other=0) < C_BLOCK:
            pass

        localmax_mat = tl.load(
            max_cache_ptr + offs_m[:, None] * stride_max_s + block_idx[None, :] * stride_max_c,
            mask=(offs_m[:, None] < s) & (block_idx[None, :] < C_BLOCK),
            other=-1e9,
        )
        globalmax = tl.max(localmax_mat, axis=1)
        factor = tl.exp(localmax_mat - globalmax[:, None])

        sumexp_mat = tl.load(
            sum_cache_ptr + offs_m[:, None] * stride_sum_s + block_idx[None, :] * stride_sum_c,
            mask=(offs_m[:, None] < s) & (block_idx[None, :] < C_BLOCK),
            other=0.0,
        )
        globalsum = tl.sum(sumexp_mat * factor, axis=1)

        qk = qk * tl.exp(localmax - globalmax)[:, None]
        qk_softmax = tl.div_rn(qk, globalsum[:, None])

        # qk_softmax @ V
        if pid_n == 0:
            tl.atomic_xchg(semaphore_ptr, 0)
        while tl.load(semaphore_ptr, mask=True, other=0) > 0:
            pass

        for idx in range(0, C_BLOCK + D_BLOCK - 1):
            # Clean the semaphore
            local_sense = tl.load(sense_rev_ptr, mask=True, other=0)
            num_proc = tl.atomic_add(semaphore_ptr, 1)
            if num_proc == C_BLOCK - 1:
                tl.atomic_xchg(semaphore_ptr, 0)
                tl.atomic_xchg(sense_rev_ptr, 1 - local_sense)
            else:
                while tl.load(sense_rev_ptr, mask=True, other=0) == local_sense:
                    pass

            d_idx = idx - pid_n 
            if d_idx >= 0 and d_idx < D_BLOCK:
                offs_d = d_idx * BLOCK_D + tl.arange(0, BLOCK_D)
                v_mat = tl.load(
                    v_ptr + offs_n[:, None] * stride_v_s + offs_d[None, :] * stride_v_d,
                    mask=(offs_n[:, None] < s) & (offs_d[None, :] < d),
                    other=0.0,
                )
                prev_mat_sum = tl.load(
                    output_rep_ptr + offs_m[:, None] * stride_output_s + offs_d[None, :] * stride_output_d,
                    mask=(offs_m[:, None] < s) & (offs_d[None, :] < d),
                    other=0.0,
                )
                curr_mat_sum = prev_mat_sum + tl.dot(qk_softmax, v_mat, input_precision="ieee")
                # curr_mat_sum = tl.cast(curr_mat_sum, DTYPE) # Downcast to old datatype
                tl.store(
                    output_rep_ptr + offs_m[:, None] * stride_output_s + offs_d[None, :] * stride_output_d,
                    curr_mat_sum,
                    mask=(offs_m[:, None] < s) & (offs_d[None, :] < d),
                )
        

def _universal_attention_fwd(q, k, v, src, dest):
    b, n, s, d = q.shape
    _, n_kv, _, _ = k.shape
    assert n % n_kv == 0, "n needs to be divisible by n_kv"
    rep = n // n_kv
    device = q.device
    dtype_flag = tl.float16 if q.dtype == torch.float16 else tl.float32
    
    BLOCK_C = 16
    BLOCK_D = 16
    C_BLOCK = triton.cdiv(s, BLOCK_C)
    D_BLOCK = triton.cdiv(d, BLOCK_D)

    output = torch.empty(b, n, s, d, dtype=q.dtype, device=device)
    
    spinlock = torch.zeros((b, n_kv, C_BLOCK), device=device, dtype=torch.int32)
    semaphore = torch.zeros((b, n_kv, C_BLOCK), device=device, dtype=torch.int32)
    sense_rev = torch.zeros((b, n_kv, C_BLOCK), device=device, dtype=torch.int32)
    cum_cache = torch.empty((b, n_kv, s), device=device, dtype=torch.float32)
    max_cache = torch.empty((b, n_kv, s, C_BLOCK), device=device, dtype=torch.float32)
    sum_cache = torch.empty((b, n_kv, s, C_BLOCK), device=device, dtype=torch.float32)

    debug = torch.empty((b, n_kv, s, s), device=device, dtype=torch.float32)
    
    # grid = lambda META: (b * n_kv, triton.cdiv(s, META['BLOCK_C']), triton.cdiv(s, META['BLOCK_C']))
    grid = (b * n_kv, C_BLOCK, C_BLOCK)

    _universal_attention_fwd_kernel[grid](
        q, k, v, src, dest, output, 
        spinlock, semaphore, sense_rev, 
        cum_cache, max_cache, sum_cache, 
        debug,
        b, n_kv, rep, s, d, 
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),                                 # (b, n, s, d)
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),                                 # (b, n_kv, s, d)
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),                                 # (b, n_kv, s, d)
        src.stride(0), src.stride(1), src.stride(2),                                        # (b, n_kv, s)
        dest.stride(0), dest.stride(1), dest.stride(2),                                     # (b, n_kv, s)
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),             # (b, n, s, d)
        spinlock.stride(0), spinlock.stride(1), spinlock.stride(2),                         # (b, n_kv, C_BLOCK)
        semaphore.stride(0), semaphore.stride(1), semaphore.stride(2),                      # (b, n_kv, C_BLOCK)
        sense_rev.stride(0), sense_rev.stride(1), sense_rev.stride(2),                      # (b, n_kv, C_BLOCK)
        cum_cache.stride(0), cum_cache.stride(1), cum_cache.stride(2),                      # (b, n_kv, s)
        max_cache.stride(0), max_cache.stride(1), max_cache.stride(2), max_cache.stride(3), # (b, n_kv, s, C_BLOCK)
        sum_cache.stride(0), sum_cache.stride(1), sum_cache.stride(2), sum_cache.stride(3), # (b, n_kv, s, C_BLOCK)
        debug.stride(0), debug.stride(1), debug.stride(2), debug.stride(3),                 # For debugging
        BLOCK_C=BLOCK_C, BLOCK_D=BLOCK_D, 
        C_BLOCK=C_BLOCK, D_BLOCK=D_BLOCK,
        DTYPE=dtype_flag,
    )
    del spinlock, semaphore, sense_rev, cum_cache, max_cache, sum_cache
    return output#, debug

'''
#################################
#     Ground Truth Autograd     #
#################################
'''
def universal_attention_forward(q, k, v, src, dest):
    b, h, l, d = k.shape
    _, nh, _, _ = q.shape
    c = 16
    n = l // c
    r = nh // h

    kc = k.view(b, h, n, c, d)
    vc = v.view(b, h, n, c, d)  
    xq = q.view(b, h, r, l, d)
    static_src = src.view(b, h, n, c)
    static_dest = dest.view(b, h, l)

    b,h,r,l,d = xq.shape
    _,_,n,c,_ = kc.shape

    mask = torch.ones(c,l, dtype=torch.bool, device=xq.device)
    out = torch.empty(b,h,r,l,d,n, dtype=xq.dtype, device=xq.device)
    denom = torch.empty(b,h,r,l,n, dtype=xq.dtype, device=xq.device)
    static_src = static_src.pow(1/3)
    static_dest = static_dest.pow(1/3)

    affinity_collect = torch.empty(b,h,c,l,n, dtype=xq.dtype, device=xq.device)

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

        # affinity_collect[...,i] = affinity
        # Perform actual attention operation
        score = k_.unsqueeze(2).matmul(xq.transpose(-1,-2)).add(affinity.unsqueeze(2))  # b h r c l
        denom_ = score.logsumexp(dim=-2)  # b h r l
        out_ = score.transpose(-1,-2).softmax(dim=-1).to(dtype=xq.dtype).matmul(v_.unsqueeze(2))  # b h r l d

        out[...,i] = out_
        denom[...,i] = denom_
    output = out.mul(denom.softmax(dim=-1).unsqueeze(-2)).sum(-1)
    # return affinity_collect
    return output.reshape(b, nh, l, d)

'''
#######################################
#     Backward Kernel & Interface     #
#######################################
'''
# @triton.autotune(
#     configs=[
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=8),
#         triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
#         triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
#         triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=2),
#     ],
#     key=['hdim', 'dstate', 'chunk_size'],
# )
# @triton.jit
# def _universal_attention_bwd_kernel(
#     # Pointers to matrices
#     # Matrix dimensions
#     # Strides
#     # Meta-parameters
# ):
#     pass

def _universal_attention_bwd(kc, vc, xq, static_src, static_dest, dout, ddenom):
    return None, None, None, None, None

if __name__ == "__main__":
    b, n_kv, rep, s, d = 2, 4, 8, 64, 64
    q = torch.rand((b, n_kv * rep, s, d), device='cuda', dtype=torch.float32)
    k = torch.rand((b, n_kv, s, d), device='cuda', dtype=torch.float32)
    v = torch.rand((b, n_kv, s, d), device='cuda', dtype=torch.float32)
    src = torch.rand((b, n_kv, s), device='cuda', dtype=torch.float32)
    dest = torch.rand((b, n_kv, s), device='cuda', dtype=torch.float32)

    # _, affinity = _universal_attention_fwd(q, k, v, src, dest)
    # affinity_ref = universal_attention_forward(q, k, v, src, dest)
    # affinity_ref = torch.permute(affinity_ref, (0, 1, 4, 2, 3)).reshape(b, n_kv, s, s)
    output = _universal_attention_fwd(q, k, v, src, dest)
    output_ref = universal_attention_forward(q, k, v, src, dest)
    # affinity_ref = torch.permute(affinity_ref, (0, 1, 4, 2, 3)).reshape(b, n_kv, s, s)

    count = 0
    for arr in (output, output_ref):
        arr = arr.cpu().numpy().reshape(-1, arr.shape[-1])
        # print(
        #     np.array2string(
        #         affinity.cpu().numpy(),
        #         precision=2,
        #         suppress_small=True,
        #         max_line_width=1200
        #     )
        # )
        np.savetxt(f"output{count}.csv", arr, delimiter=",", fmt="%.6f")   
        count += 1

    # output = _universal_attention_fwd(q, k, v, src, dest)
    # output_ref = universal_attention_forward(q, k, v, src, dest)
    diff = (output - output_ref).abs()
    print("Max abs diff:", diff.max().item())

    torch.testing.assert_close(output, output_ref)

import torch
import triton
import triton.language as tl

configs = [
    triton.Config({}, num_stages=stages, num_warps=warps) \
    for stages in [2, 3, 4, 5]\
    for warps in [2, 4, 8, 16]\
]

'''
######################################
#     Forward Kernel & Interface     #
######################################
'''
@triton.autotune(
    configs=configs,
    key=['s', 'd'],
)
@triton.jit
def _universal_attention_fwd_kernel(
    # Pointers to matrices
    q_ptr, k_ptr, v_ptr, src_ptr, dest_ptr, output_ptr, 
    sema_ptr, cache_ptr, 
    # Matrix dimensions
    b, n_kv, rep, s, d, 
    # Strides
    stride_q_b, stride_q_n, stride_q_s, stride_q_d, 
    stride_k_b, stride_k_n_kv, stride_k_s, stride_k_d, 
    stride_v_b, stride_v_n_kv, stride_v_s, stride_v_d,
    stride_src_b, stride_src_n_kv, stride_src_s, 
    stride_dest_b, stride_dest_n_kv, stride_dest_s, 
    stride_output_b, stride_output_n, stride_output_s, stride_output_d, 
    stride_sema_b, stride_sema_n_kv, stride_sema_s,
    stride_cache_b, stride_cache_n_kv, stride_cache_s,
    # Meta-parameters
    BLOCK_C: tl.constexpr, BLOCK_D: tl.constexpr, 
    DTYPE: tl.constexpr,
):
    pid_b_n_kv = tl.program_id(0)
    pid_b = pid_b_n_kv // n_kv
    pid_n_kv = pid_b_n_kv % n_kv
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    offs_m = pid_m * BLOCK_C + tl.arange(0, BLOCK_C)
    offs_n = pid_n * BLOCK_C + tl.arange(0, BLOCK_C)
    
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

        # Use ieee to use fp32, otherwise the default would be tf32
        affinity += tl.dot(k_mat, tl.trans(kt_mat), input_precision="ieee")
    
    # .relu()
    affinity = tl.maximum(affinity, 0.0)

    # .pow(2/3)
    affinity = tl.exp2(tl.log2(affinity) * 2.0 / 3.0)

    # * static_src_.pow(1/3).unsqueeze(-1) * static_dest.pow(1/3).unsqueeze(-2)
    src_ptr += pid_b * stride_src_b + pid_n_kv * stride_src_n_kv
    dest_ptr += pid_b * stride_dest_b + pid_n_kv * stride_dest_n_kv

    src_mat  = tl.load(src_ptr + offs_m * stride_src_s, mask=offs_m < s, other=0.0)
    src_mat = tl.cast(src_mat, tl.float32)
    src_mat = tl.exp2(tl.log2(src_mat) / 3.0)

    dest_mat = tl.load(dest_ptr + offs_n * stride_dest_s, mask=offs_n < s, other=0.0)
    dest_mat = tl.cast(dest_mat, tl.float32)
    dest_mat = tl.exp2(tl.log2(dest_mat) / 3.0)

    affinity = affinity * src_mat[:, None] * dest_mat[None, :]

    # torch.log1p(affinity.clamp(min=0, max=1-1e-6).neg())
    affinity = tl.clamp(affinity, 0.0, 1.0 - 1e-6)
    affinity = tl.log(1.0 - affinity) 

    # .triu(1)
    triu_mask = (offs_n[None, :] >= (offs_m[:, None] - 1))
    affinity = tl.where(triu_mask, affinity, 0.0)

    # .cumsum(3)
    sem_ptr += pid_b * stride_sema_b + pid_n_kv * stride_sema_n_kv + pid_m * stride_sema_s
    cache_ptr += pid_b * stride_cache_b + pid_n_kv * stride_cache_n_kv 
    
    affinity = tl.cumsum(affinity, axis=1) # local cumsum
    curr_sum = tl.sum(A_mat, axis=1, keep_dims=False) # put the sum into the cache

    # Make sure the sum is passed sequentially
    while tl.load(sem_ptr, mask=True, other=0) < pid_n:
        pass

    prev_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)
    if pid_n > 0:
        prev_sum = tl.load(
            cache_ptr + offs_m * stride_cache_s, 
            mask=offs_m < m, 
            other=0.0
        ) 
    tl.store(cache_ptr + offs_m * stride_cache_s, curr_sum + prev_sum, mask=offs_m < m)

    tl.atomic_add(sem_ptr, 1)

    affinity = affinity + prev_sum[:, None]

    # .masked_fill(mask.tril(-1), -1e12)
    tril_mask = (offs_n[None, :] <= (offs_m[:, None] - 1))
    affinity = tl.where(tril_mask, -1e12, affinity)

    # Affinity compute completed here

    # k_.unsqueeze(2).matmul(xq.transpose(-1,-2)).add(affinity.unsqueeze(2))
    q_ptr += pid_b * stride_q_b + pid_n_kv * rep * stride_q_n
    v_ptr += pid_b * stride_v_b + pid_n_kv * rep * stride_v_n
    output_ptr += pid_b * stride_output_b + pid_n_kv * rep * stride_output_n

    # Haochen: 3D mat mul is supported by triton, but not by llvm in general, so it is not used here. 
    for r in range(0, rep):
        # k @ q^T
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
                q_ptr + r * stride_q_n + offs_n[:, None] * stride_q_s + offs_k[None, :] * stride_q_d, 
                mask=(offs_n[:, None] < s) & (offs_k[None, :] < d), 
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
            max_cache_ptr + offs_m * stride_max_m + pid_n * stride_max_n, 
            localmax, 
            mask=(offs_m < m)
        )

        qk = tl.exp(qk - localmax[:, None])
        qk_sumexp = tl.sum(qk, axis=1)
        tl.store(
            sum_cache_ptr + offs_m * stride_sum_m + pid_n * stride_sum_n, 
            qk_sumexp, 
            mask=(offs_m < m)
        )

        tl.atomic_add(semaphore_ptr, 1)
        # Don't use atomic read here, or it will prevent other atomic operations
        while tl.load(semaphore_ptr, mask=True, other=0) < N_BLOCK:
            pass

        localmax_mat = tl.load(
            max_cache_ptr + offs_m[:, None] * stride_max_m + block_idx[None, :] * stride_max_n,
            mask=(offs_m[:, None] < m) & (block_idx[None, :] < N_BLOCK),
            other=-1e9,
        )
        globalmax = tl.max(localmax_mat, axis=1)
        factor = tl.exp(localmax_mat - globalmax[:, None])

        sumexp_mat = tl.load(
            sum_cache_ptr + offs_m[:, None] * stride_sum_m + block_idx[None, :] * stride_sum_n,
            mask=(offs_m[:, None] < m) & (block_idx[None, :] < N_BLOCK),
            other=0.0,
        )
        globalsum = tl.sum(sumexp_mat * factor, axis=1)

        qk = qk * tl.exp(localmax - globalmax)[:, None]
        qk_softmax = tl.div_rn(qk, globalsum[:, None])

        # qk_softmax @ V
        for c in range(0, tl.cdiv(s, BLOCK_C)):
            if c == 
            c_offset = c * BLOCK_C
            # qk_softmax @ v: Load chunks (BLOCK_C, BLOCK_C) @ (BLOCK_C, BLOCK_D)
            
        for d_offset in range(0, d, BLOCK_D):
            offs_d = d_offset + tl.arange(0, BLOCK_D)

            k_mat = tl.load(
                k_ptr + offs_m[:, None] * stride_k_s + offs_d[None, :] * stride_k_d, 
                mask=(offs_m[:, None] < s) & (offs_d[None, :] < d), 
                other=0.0,
            )
            k_mat = tl.cast(k_mat, tl.float32)

            q_mat = tl.load(
                q_ptr + r * stride_q_n + offs_n[:, None] * stride_q_s + offs_k[None, :] * stride_q_d, 
                mask=(offs_n[:, None] < s) & (offs_k[None, :] < d), 
                other=0.0,
            )
            q_mat = tl.cast(q_mat, tl.float32)

            # Use ieee to use fp32, otherwise the default would be tf32
            qk += tl.dot(k_mat, tl.trans(q_mat), input_precision="ieee")

        # I'm here! The only thing left is Softmax(...) @ V

        acc = tl.cast(acc, DTYPE) # Downcast to old datatype

        tl.store(
            output_ptr + r * stride_output_n + offs_m[:, None] * stride_output_s + offs_n[None, :] * stride_output_d,
            acc,
            mask=(offs_m[:, None] < i) & (offs_n[None, :] < j),
        )
    

def _universal_attention_fwd(q, k, v, src, dest):
    b, n, s, d = q.shape
    _, n_kv, _, _ = k.shape
    assert n % n_kv == 0, "n needs to be divisible by n_kv"
    rep = n // n_kv
    device = q.device
    dtype_flag = tl.float16 if A.dtype == torch.float16 else tl.float32
    
    output = torch.empty(b, n, s, d, dtype=q.dtype, device=device)
    semaphore = torch.zeros(b, n_kv, s, dtype=torch.int32, device=device)
    cs_cache = torch.empty(b, n_kv, s, dtype=torch.float32, device=device)

    grid = lambda META: (b * n_kv, triton.cdiv(s, META['BLOCK_C']), triton.cdiv(s, META['BLOCK_C']))

    _universal_attention_fwd_kernel[grid](
        q, k, v, src, dest, output, semaphore, cs_cache,
        b, n_kv, rep, s, d, 
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),                         # (b, n, s, d)
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),                         # (b, n_kv, s, d)
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),                         # (b, n_kv, s, d)
        src.stride(0), src.stride(1), src.stride(2),                                # (b, n_kv, s)
        dest.stride(0), dest.stride(1), dest.stride(2),                             # (b, n_kv, s)
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),     # (b, n, s, d)
        semaphore.stride(0), semaphore.stride(1), semaphore.stride(2),              # (b, n_kv, s)
        cs_cache.stride(0), cs_cache.stride(1), cs_cache.stride(2),                 # (b, n_kv, s)
        DTYPE=dtype_flag,
    )

    return output


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
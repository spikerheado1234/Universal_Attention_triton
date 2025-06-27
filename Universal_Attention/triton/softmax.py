import triton
import triton.language as tl
import torch
import time

configs = [
    triton.Config({}, num_stages=stages, num_warps=warps) \
    for stages in [2, 3, 4]\
    for warps in [2, 4, 8]\
]

@triton.autotune(
    configs=configs,
    key=['m', 'n'],
)
@triton.jit
def softmax_kernel(
    A_ptr, B_ptr, semaphore_ptr, max_cache_ptr, sum_cache_ptr, 
    b, n_kv, m, n, 
    stride_a_b, stride_a_n_kv, stride_a_m, stride_a_n,
    stride_b_b, stride_b_n_kv, stride_b_m, stride_b_n, 
    stride_sem_b, stride_sem_n_kv, stride_sem_m,
    stride_max_b, stride_max_n_kv, stride_max_m, stride_max_n, 
    stride_sum_b, stride_sum_n_kv, stride_sum_m, stride_sum_n, 
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, N_BLOCK: tl.constexpr,
):
    pid_b_n_kv = tl.program_id(0)
    pid_b = pid_b_n_kv // n_kv
    pid_n_kv = pid_b_n_kv % n_kv
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    block_idx = tl.arange(0, N_BLOCK)

    A_ptr += pid_b * stride_a_b + pid_n_kv * stride_a_n_kv
    B_ptr += pid_b * stride_b_b + pid_n_kv * stride_b_n_kv

    semaphore_ptr += pid_b * stride_sem_b + pid_n_kv * stride_sem_n_kv + pid_m * stride_sem_m
    max_cache_ptr += pid_b * stride_max_b + pid_n_kv * stride_max_n_kv
    sum_cache_ptr += pid_b * stride_sum_b + pid_n_kv * stride_sum_n_kv
    
    A_mat = tl.load(
        A_ptr + offs_m[:, None] * stride_a_m + offs_n[None, :] * stride_a_n,
        mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
        other=-1e9,
    )
    # Upcast to fp32 to align with pytorch results
    A_mat = tl.cast(A_mat, tl.float32)

    localmax = tl.max(A_mat, axis=1)
    # store local max into cache
    tl.store(max_cache_ptr + offs_m * stride_max_m + pid_n * stride_max_n, localmax, mask=(offs_m < m))

    A_mat = tl.exp(A_mat - localmax[:, None])
    A_sumexp = tl.sum(A_mat, axis=1)
    # store local sum of exp into cache
    tl.store(sum_cache_ptr + offs_m * stride_sum_m + pid_n * stride_sum_n, A_sumexp, mask=(offs_m < m))
    
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

    A_mat = A_mat * tl.exp(localmax - globalmax)[:, None]
    A_softmax = tl.div_rn(A_mat, globalsum[:, None])

    tl.store(
        B_ptr + offs_m[:, None] * stride_b_m + offs_n[None, :] * stride_b_n,
        A_softmax,
        mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
    )

def softmax_triton(A: torch.Tensor) -> torch.Tensor:
    b, n_kv, m, n = A.shape

    BLOCK_M = 64
    BLOCK_N = 64
    M_BLOCK = triton.cdiv(m, BLOCK_M)
    N_BLOCK = triton.cdiv(n, BLOCK_N)

    if A.stride(-1) != 1:
        A = A.contiguous()

    B = torch.empty((b, n_kv, m, n), device=A.device, dtype=A.dtype)
    semaphore = torch.zeros((b, n_kv, M_BLOCK), device=A.device, dtype=torch.int32)
    max_cache = torch.empty((b, n_kv, m, N_BLOCK), device=A.device, dtype=A.dtype)
    sum_cache = torch.empty((b, n_kv, m, N_BLOCK), device=A.device, dtype=A.dtype)

    # grid = lambda META: (b * n_kv, triton.cdiv(m, META['BLOCK_M']), triton.cdiv(n, META['BLOCK_N']))
    grid = (b * n_kv, M_BLOCK, N_BLOCK)

    softmax_kernel[grid](
        A, B, semaphore, max_cache, sum_cache, 
        b, n_kv, m, n, 
        A.stride(0), A.stride(1), A.stride(2), A.stride(3), 
        B.stride(0), B.stride(1), B.stride(2), B.stride(3),
        semaphore.stride(0), semaphore.stride(1), semaphore.stride(2), 
        max_cache.stride(0), max_cache.stride(1), max_cache.stride(2), max_cache.stride(3),
        sum_cache.stride(0), sum_cache.stride(1), sum_cache.stride(2), sum_cache.stride(3),
        BLOCK_M, BLOCK_N, N_BLOCK,
    )
        
    return B

if __name__ == "__main__":
    b, n_kv, m, n = 2, 4, 16384, 16384
    A = torch.randn(b, n_kv, m, n, device='cuda', dtype=torch.float32) * 10
    
    # Warm up first
    _ = softmax_triton(A)
    torch.cuda.synchronize()

    start_time = time.time()
    B = softmax_triton(A)
    print(f"Triton kernel time: {time.time() - start_time}")

    start_time = time.time()
    B_ref = A.softmax(dim=-1)
    print(f"Pytorch kernel time: {time.time() - start_time}")

    print("Max error:", (B.float() - B_ref.float()).abs().max())

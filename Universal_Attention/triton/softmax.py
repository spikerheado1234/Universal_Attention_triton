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

@triton.autotune(
    configs=configs,
    key=['m', 'n', 'k'],
)
@triton.jit
def softmax_matmul_kernel_v1(
    A_ptr, B_ptr, C_ptr, 
    semaphore_ptr, sense_rev_ptr, max_cache_ptr, sum_cache_ptr, 
    b, n_kv, m, n, k, 
    stride_a_b, stride_a_n_kv, stride_a_m, stride_a_n,
    stride_b_b, stride_b_n_kv, stride_b_n, stride_b_k, 
    stride_c_b, stride_c_n_kv, stride_c_m, stride_c_k, 
    stride_sem_b, stride_sem_n_kv, stride_sem_m,
    stride_rev_b, stride_rev_n_kv, stride_rev_m,
    stride_max_b, stride_max_n_kv, stride_max_m, stride_max_n, 
    stride_sum_b, stride_sum_n_kv, stride_sum_m, stride_sum_n, 
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, 
    N_BLOCK: tl.constexpr, K_BLOCK: tl.constexpr, 
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
    C_ptr += pid_b * stride_c_b + pid_n_kv * stride_c_n_kv

    semaphore_ptr += pid_b * stride_sem_b + pid_n_kv * stride_sem_n_kv + pid_m * stride_sem_m
    sense_rev_ptr += pid_b * stride_rev_b + pid_n_kv * stride_rev_n_kv + pid_m * stride_rev_m
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

    # Matmul after softmax
    if pid_n == 0:
        tl.atomic_xchg(semaphore_ptr, 0)
    while tl.load(semaphore_ptr, mask=True, other=0) > 0:
        pass

    for idx in range(0, N_BLOCK + K_BLOCK - 1):
        # Clean the semaphore
        local_sense = tl.load(sense_rev_ptr, mask=True, other=0)
        num_proc = tl.atomic_add(semaphore_ptr, 1)
        if num_proc == N_BLOCK - 1:
            tl.atomic_xchg(semaphore_ptr, 0)
            tl.atomic_xchg(sense_rev_ptr, 1 - local_sense)
        else:
            while tl.load(sense_rev_ptr, mask=True, other=0) == local_sense:
                pass

        k_idx = idx - pid_n 
        if k_idx >= 0 and k_idx < K_BLOCK:
            offs_k = k_idx * BLOCK_K + tl.arange(0, BLOCK_K)
            B_mat = tl.load(
                B_ptr + offs_n[:, None] * stride_b_n + offs_k[None, :] * stride_b_k,
                mask=(offs_n[:, None] < n) & (offs_k[None, :] < k),
                other=0.0,
            )
            prev_sum = tl.load(
                C_ptr + offs_m[:, None] * stride_c_m + offs_k[None, :] * stride_c_k,
                mask=(offs_m[:, None] < m) & (offs_k[None, :] < k),
                other=0.0,
            )
            curr_sum = prev_sum + tl.dot(A_softmax, B_mat, input_precision="ieee")
            tl.store(
                C_ptr + offs_m[:, None] * stride_c_m + offs_k[None, :] * stride_c_k,
                curr_sum,
                mask=(offs_m[:, None] < m) & (offs_k[None, :] < k),
            )

def softmax_matmul_triton(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    b, n_kv, m, n = A.shape
    b, n_kv, n, k = B.shape

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 64
    M_BLOCK = triton.cdiv(m, BLOCK_M)
    N_BLOCK = triton.cdiv(n, BLOCK_N)
    K_BLOCK = triton.cdiv(k, BLOCK_K)

    if A.stride(-1) != 1:
        A = A.contiguous()

    C = torch.zeros((b, n_kv, m, k), device=A.device, dtype=A.dtype)
    
    semaphore = torch.zeros((b, n_kv, M_BLOCK), device=A.device, dtype=torch.int32)
    sense_rev = torch.zeros((b, n_kv, M_BLOCK), device=A.device, dtype=torch.int32)
    max_cache = torch.empty((b, n_kv, m, N_BLOCK), device=A.device, dtype=A.dtype)
    sum_cache = torch.empty((b, n_kv, m, N_BLOCK), device=A.device, dtype=A.dtype)
    
    # grid = lambda META: (b * n_kv, triton.cdiv(m, META['BLOCK_M']), triton.cdiv(n, META['BLOCK_N']))
    grid = (b * n_kv, M_BLOCK, N_BLOCK)

    softmax_matmul_kernel_v1[grid](
        A, B, C, 
        semaphore, sense_rev, max_cache, sum_cache, 
        b, n_kv, m, n, k, 
        A.stride(0), A.stride(1), A.stride(2), A.stride(3), 
        B.stride(0), B.stride(1), B.stride(2), B.stride(3),
        C.stride(0), C.stride(1), C.stride(2), C.stride(3),
        semaphore.stride(0), semaphore.stride(1), semaphore.stride(2), 
        sense_rev.stride(0), sense_rev.stride(1), sense_rev.stride(2),
        max_cache.stride(0), max_cache.stride(1), max_cache.stride(2), max_cache.stride(3),
        sum_cache.stride(0), sum_cache.stride(1), sum_cache.stride(2), sum_cache.stride(3),
        BLOCK_M, BLOCK_N, BLOCK_K, N_BLOCK, K_BLOCK,
    )
        
    return C

if __name__ == "__main__":
    # test = "softmax"
    test = "softmax_matmul"

    if test == "softmax":
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

    elif test == "softmax_matmul":
        b, n_kv, m, n, k = 2, 4, 8192, 16384, 64
        A = torch.randn(b, n_kv, m, n, device='cuda', dtype=torch.float32) * 10
        B = torch.randn(b, n_kv, n, k, device='cuda', dtype=torch.float32) * 10
        
        # Warm up first
        _ = softmax_matmul_triton(A, B)
        torch.cuda.synchronize()

        start_time = time.time()
        C = softmax_matmul_triton(A, B)
        print(f"Triton kernel time: {time.time() - start_time}")

        start_time = time.time()
        C_ref = A.softmax(dim=-1) @ B
        print(f"Pytorch kernel time: {time.time() - start_time}")

        print("Max error:", (C.float() - C_ref.float()).abs().max())
        

import triton
import triton.language as tl
import torch
import time
import math

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
def cumsum_kernel(
    A_ptr, B_ptr, spinlock_ptr, cum_cache_ptr, 
    b, n_kv, m, n, 
    stride_a_b, stride_a_n_kv, stride_a_m, stride_a_n,
    stride_b_b, stride_b_n_kv, stride_b_m, stride_b_n,
    stride_spl_b, stride_spl_n_kv, stride_spl_m, 
    stride_cum_b, stride_cum_n_kv, stride_cum_m,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    DTYPE: tl.constexpr,
):
    pid_b_n_kv = tl.program_id(0)
    pid_b = pid_b_n_kv // n_kv
    pid_n_kv = pid_b_n_kv % n_kv
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    A_ptr += pid_b * stride_a_b + pid_n_kv * stride_a_n_kv
    B_ptr += pid_b * stride_b_b + pid_n_kv * stride_b_n_kv
    
    spinlock_ptr += pid_b * stride_spl_b + pid_n_kv * stride_spl_n_kv + pid_m * stride_spl_m
    cum_cache_ptr += pid_b * stride_cum_b + pid_n_kv * stride_cum_n_kv 

    A_mat = tl.load(
        A_ptr + offs_m[:, None] * stride_a_m + offs_n[None, :] * stride_a_n,
        mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
        other=0.0,
    )
    # Upcast to fp32 to align with pytorch results
    A_mat = tl.cast(A_mat, tl.float32)

    acc = tl.cumsum(A_mat, axis=-1)
    curr_sum = tl.sum(A_mat, axis=-1, keep_dims=False) # put the sum into the cache

    while tl.atomic_add(spinlock_ptr, 0, sem="acquire") < pid_n:
        pass

    prev_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)
    if pid_n > 0:
        prev_sum = tl.load(
            cum_cache_ptr + offs_m * stride_cum_m, 
            mask=offs_m < m, 
            other=0.0
        ) 
    tl.store(
        cum_cache_ptr + offs_m * stride_cum_m,
        curr_sum + prev_sum,
        mask=offs_m < m, 
    )
    tl.atomic_add(spinlock_ptr, 1, sem="release")

    acc = acc + prev_sum[:, None]

    # Downcast to original datatype
    acc = tl.cast(acc, DTYPE)
        
    tl.store(
        B_ptr + offs_m[:, None] * stride_b_m + offs_n[None, :] * stride_b_n,
        acc,
        mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
    )



def cumsum_triton(A: torch.Tensor) -> torch.Tensor:
    b, n_kv, m, n = A.shape
    dtype_flag = tl.float16 if A.dtype == torch.float16 else tl.float32

    BLOCK_M = 64
    BLOCK_N = 64
    M_BLOCK = triton.cdiv(m, BLOCK_M)
    N_BLOCK = triton.cdiv(n, BLOCK_N)

    if A.stride(-1) != 1:
        A = A.contiguous()

    B = torch.empty((b, n_kv, m, n), device=A.device, dtype=A.dtype)
    spinlock = torch.zeros((b, n_kv, M_BLOCK), device=A.device, dtype=torch.int32)
    cum_cache = torch.empty((b, n_kv, m), device=A.device, dtype=torch.float32)

    # grid = lambda META: (b * n_kv, triton.cdiv(m, META['BLOCK_M']), triton.cdiv(n, META['BLOCK_N']))
    grid = (b * n_kv, M_BLOCK, N_BLOCK)

    cumsum_kernel[grid](
        A, B, spinlock, cum_cache,
        b, n_kv, m, n, 
        A.stride(0), A.stride(1), A.stride(2), A.stride(3),
        B.stride(0), B.stride(1), B.stride(2), B.stride(3),
        spinlock.stride(0), spinlock.stride(1), spinlock.stride(2), 
        cum_cache.stride(0), cum_cache.stride(1), cum_cache.stride(2), 
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        DTYPE=dtype_flag,
    )
        
    return B

if __name__ == "__main__":
    b, n_kv, m, n = 2, 4, 16384, 16384
    runs = 1000
    dtype = torch.float32

    A = torch.randn(b, n_kv, m, n, device='cuda', dtype=dtype)

    _ = cumsum_triton(A)
    torch.cuda.synchronize()

    triton_time, torch_time = 0, 0
    for _ in range(runs):
        A = torch.randn(b, n_kv, m, n, device='cuda', dtype=dtype)

        start_time = time.time()
        _ = cumsum_triton(A)  
        triton_time += time.time() - start_time

        start_time = time.time()
        _ = torch.cumsum(A, dim=-1)
        torch_time += time.time() - start_time

        del A
    
    A = torch.randn(b, n_kv, m, n, device='cuda', dtype=dtype)
    B = cumsum_triton(A)
    B_ref = torch.cumsum(A, dim=-1)

    print(f"Triton kernel time: {triton_time}")
    print(f"Pytorch kernel time: {torch_time}")

    print("Avg error:", (B - B_ref).abs().mean())
    print("Max error:", (B - B_ref).abs().max())

import triton
import triton.language as tl
import torch
import time
import math

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=2),
    ],
    key=['m', 'n'],
)
@triton.jit
def cumsum_kernel(
    A_ptr, A_cs_ptr, sem_ptr, cs_cache_ptr, 
    b, n_kv, m, n, 
    stride_ab, stride_an_kv, stride_am, stride_an,
    stride_acsb, stride_acsn_kv, stride_acsm, stride_acsn,
    stride_semab, stride_seman_kv, stride_semam, 
    stride_ccb, stride_ccn_kv, stride_ccm,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    DTYPE: tl.constexpr,
):
    pid_b_n_kv = tl.program_id(0)
    pid_b = pid_b_n_kv // n_kv
    pid_n_kv = pid_b_n_kv % n_kv
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    A_ptr += pid_b * stride_ab + pid_n_kv * stride_an_kv
    A_cs_ptr += pid_b * stride_acsb + pid_n_kv * stride_acsn_kv
    sem_ptr += pid_b * stride_semab + pid_n_kv * stride_seman_kv + pid_m * stride_semam
    cs_cache_ptr += pid_b * stride_ccb + pid_n_kv * stride_ccn_kv #+ offs_m

    A_mat = tl.load(
        A_ptr + offs_m[:, None] * stride_am + offs_n[None, :] * stride_an,
        mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
        other=0.0,
    )
    # Upcast to fp32 to align with pytorch results
    A_mat = tl.cast(A_mat, tl.float32)

    acc = tl.cumsum(A_mat, axis=-1)
    curr_sum = tl.sum(A_mat, axis=-1, keep_dims=False) # put the sum into the cache

    while tl.atomic_add(sem_ptr, 0) < pid_n:
        pass

    prev_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)
    if pid_n > 0:
        prev_sum = tl.load(
            cs_cache_ptr + offs_m * stride_ccm, 
            mask=offs_m < m, 
            other=0.0
        ) 
    tl.store(
        cs_cache_ptr + offs_m * stride_ccm,
        curr_sum + prev_sum,
        mask=offs_m < m, 
    )
    tl.atomic_add(sem_ptr, 1)

    acc = acc + prev_sum[:, None]

    # Downcast to original datatype
    acc = tl.cast(acc, DTYPE)
        
    tl.store(
        A_cs_ptr + offs_m[:, None] * stride_acsm + offs_n[None, :] * stride_acsn,
        acc,
        mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
    )



def efficient_cumsum(A: torch.Tensor) -> torch.Tensor:
    assert A.device.type == 'cuda', "This implementation requires CUDA tensors"

    b, n_kv, m, n = A.shape
    torch.cuda.set_device(A.device)
    dtype_flag = tl.float16 if A.dtype == torch.float16 else tl.float32

    if A.stride(-1) != 1:
        A = A.contiguous()

    A_cs = torch.empty((b, n_kv, m, n), device=A.device, dtype=A.dtype)
    semaphore = torch.zeros(b, n_kv, m, device=A.device, dtype=torch.int32)
    cs_cache = torch.empty((b, n_kv, m), device=A.device, dtype=torch.float32)

    grid = lambda META: (b * n_kv, triton.cdiv(m, META['BLOCK_M']), triton.cdiv(n, META['BLOCK_N']))

    cumsum_kernel[grid](
        A, A_cs, semaphore, cs_cache,
        b, n_kv, m, n, 
        A.stride(0), A.stride(1), A.stride(2), A.stride(3),
        A_cs.stride(0), A_cs.stride(1), A_cs.stride(2), A_cs.stride(3),
        semaphore.stride(0), semaphore.stride(1), semaphore.stride(2), 
        cs_cache.stride(0), cs_cache.stride(1), cs_cache.stride(2), 
        DTYPE=dtype_flag,
    )
        
    return A_cs

if __name__ == "__main__":
    b, n_kv, m, n = 2, 4, 16384, 16384
    runs = 1000
    dtype = torch.float32

    A = torch.randn(b, n_kv, m, n, device='cuda', dtype=dtype)

    _ = efficient_cumsum(A)
    torch.cuda.synchronize()

    triton_time, torch_time = 0, 0
    for _ in range(runs):
        A = torch.randn(b, n_kv, m, n, device='cuda', dtype=dtype)

        start_time = time.time()
        _ = efficient_cumsum(A)  
        triton_time += time.time() - start_time

        start_time = time.time()
        _ = torch.cumsum(A, dim=-1)
        torch_time += time.time() - start_time

        del A
    
    A = torch.randn(b, n_kv, m, n, device='cuda', dtype=dtype)
    A_cs = efficient_cumsum(A)
    A_cs_ref = torch.cumsum(A, dim=-1)

    print(f"Triton kernel time: {triton_time}")
    print(f"Pytorch kernel time: {torch_time}")

    print("Avg error:", (A_cs - A_cs_ref).abs().mean())
    print("Max error:", (A_cs - A_cs_ref).abs().max())

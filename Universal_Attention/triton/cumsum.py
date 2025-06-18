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
    A_ptr, A_cs_ptr,
    b, n_kv, m, n, 
    stride_ab, stride_an_kv, stride_am, stride_an,
    stride_acsb, stride_acsn_kv, stride_acsm, stride_acsn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
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

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    A_mat = tl.load(
        A_ptr + offs_m[:, None] * stride_am + offs_n[None, :] * stride_an,
        mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
        other=0.0,
    )

    acc = tl.cumsum(A_mat, axis=-1)

    tl.store(
        A_cs_ptr + offs_m[:, None] * stride_acsm + offs_n[None, :] * stride_acsn,
        acc,
        mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
    )

def efficient_cumsum(A: torch.Tensor) -> torch.Tensor:
    assert A.device.type == 'cuda', "This implementation requires CUDA tensors"

    b, n_kv, m, n = A.shape
    torch.cuda.set_device(A.device)

    if A.stride(-1) != 1:
        A = A.contiguous()

    A_cs = torch.empty((b, n_kv, m, n), device=A.device, dtype=A.dtype)

    grid = lambda META: (b * n_kv, triton.cdiv(m, META['BLOCK_M']), triton.cdiv(n, META['BLOCK_N']))

    cumsum_kernel[grid](
        A, A_cs,
        b, n_kv, m, n, 
        A.stride(0), A.stride(1), A.stride(2), A.stride(3),
        A_cs.stride(0), A_cs.stride(1), A_cs.stride(2), A_cs.stride(3),
    )
        
    return A_cs

if __name__ == "__main__":
    # b, n_kv, m, n = 2, 4, 128, 256
    # A = torch.randn(b, n_kv, m, n, device='cuda', dtype=torch.float16)
    A = torch.arange(1, 10, device='cuda', dtype=torch.float32).view(1, 1, 1, -1)

    _ = efficient_cumsum(A)
    torch.cuda.synchronize()

    start_time = time.time()
    A_cs = efficient_cumsum(A)
    print(f"Pytorch kernel time: {time.time() - start_time}")

    start_time = time.time()
    A_cs_ref = torch.cumsum(A, dim=-1)
    print(f"Pytorch kernel time: {time.time() - start_time}")

    print("Max error:", (A_cs.float() - A_cs_ref.float()).abs().max())



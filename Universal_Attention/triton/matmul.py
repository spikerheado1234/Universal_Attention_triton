import triton
import triton.language as tl
import torch
import time

# @triton.autotune(
#     configs=[
#         triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
#         triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=2, num_warps=8),
#         triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_stages=2, num_warps=16),
#         triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
#     ],
#     key=['S', 'D'],
# )
# @triton.jit
# def matmul_4d_kernel(
#     A_ptr, B_ptr, C_ptr,
#     BATCH, S, D,
#     stride_ab, stride_as, stride_ad,
#     stride_bb, stride_bd, stride_bs,
#     stride_cb, stride_cs, stride_cn,
#     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
# ):
#     # program axes
#     pid_batch = tl.program_id(0)
#     pid_m     = tl.program_id(1)
#     pid_n     = tl.program_id(2)

#     offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
#     offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

#     A_ptr += pid_batch * stride_ab
#     B_ptr += pid_batch * stride_bb
#     C_ptr += pid_batch * stride_cb

#     acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

#     for k_offset in range(0, D, BLOCK_K):
#         offs_k = k_offset + tl.arange(0, BLOCK_K)
#         a_ptrs = A_ptr + offs_m[:, None] * stride_as + offs_k[None, :] * stride_ad
#         a = tl.load(
#             a_ptrs,
#             mask=(offs_m[:, None] < S) & (offs_k[None, :] < D),
#             other=0.0,
#         )
#         b_ptrs = B_ptr + offs_k[:, None] * stride_bd + offs_n[None, :] * stride_bs
#         b = tl.load(
#             b_ptrs,
#             mask=(offs_k[:, None] < D) & (offs_n[None, :] < S),
#             other=0.0,
#         )
#         acc += tl.dot(a, b)

#     c_ptrs = C_ptr + offs_m[:, None] * stride_cs + offs_n[None, :] * stride_cn
#     tl.store(
#         c_ptrs,
#         acc,
#         mask=(offs_m[:, None] < S) & (offs_n[None, :] < S),
#     )

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_stages=2, num_warps=16),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
    ],
    key=['m', 'k', 'n'],
)
@triton.jit
def matmul_4d_kernel(
    A_ptr, B_ptr, C_ptr,
    b, n_kv, m, k, n, 
    stride_ab, stride_an_kv, stride_am, stride_ak,
    stride_bb, stride_bn_kv, stride_bk, stride_bn,
    stride_cb, stride_cn_kv, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # program axes
    pid_b_n_kv = tl.program_id(0)
    pid_b = pid_b_n_kv // n_kv
    pid_n_kv = pid_b_n_kv % n_kv
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    A_ptr += pid_b * stride_ab + pid_n_kv * stride_an_kv
    B_ptr += pid_b * stride_bb + pid_n_kv * stride_bn_kv
    C_ptr += pid_b * stride_cb + pid_n_kv * stride_cn_kv

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_offset in range(0, k, BLOCK_K):
        offs_k = k_offset + tl.arange(0, BLOCK_K)
        A_mat = tl.load(
            A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
            mask=(offs_m[:, None] < m) & (offs_k[None, :] < k),
            other=0.0,
        )
        B_mat = tl.load(
            B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            mask=(offs_k[:, None] < k) & (offs_n[None, :] < n),
            other=0.0,
        )
        acc += tl.dot(A_mat, B_mat)

    tl.store(
        C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        acc,
        mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
    )

def efficient_4d_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Perform A@B where:
      A: [b, n_kv, S, D]
      B: [b, n_kv, D, S]
    returning C: [b, n_kv, S, S]
    using tiled Triton kernel to chunk over S and D.
    """
    assert A.ndim == 4 and B.ndim == 4, "Inputs must be 4-D"
    b, n_kv, m, k = A.shape
    _, _, _, n = B.shape
    assert A.shape[-1] == B.shape[-2], "Dimension miss match!"

    # A_flat = A.contiguous().view(-1, S, D)
    # B_flat = B.contiguous().view(-1, D, S)
    # C_flat = torch.empty((b * n_kv, S, S), device=A.device, dtype=A.dtype)

    A = A.contiguous()
    B = B.contiguous()
    C = torch.empty((b, n_kv, m, n), device=A.device, dtype=A.dtype)

    grid = lambda META: (b * n_kv, triton.cdiv(m, META['BLOCK_M']), triton.cdiv(n, META['BLOCK_N']))

    # with torch.cuda.device('cuda'):
    #     matmul_4d_kernel[grid](
    #         A_flat, B_flat, C_flat,
    #         b * n_kv, S, D,
    #         A_flat.stride(0), A_flat.stride(1), A_flat.stride(2),
    #         B_flat.stride(0), B_flat.stride(1), B_flat.stride(2),
    #         C_flat.stride(0), C_flat.stride(1), C_flat.stride(2),
    #     )
        
    # return C_flat.view(b, n_kv, S, S)
    
    with torch.cuda.device('cuda'):
        matmul_4d_kernel[grid](
            A, B, C,
            b, n_kv, m, k, n, 
            A.stride(0), A.stride(1), A.stride(2), A.stride(3),
            B.stride(0), B.stride(1), B.stride(2), B.stride(3),
            C.stride(0), C.stride(1), C.stride(2), C.stride(3),
        )
        
    return C

if __name__ == "__main__":
    b, n_kv, m, k, n = 2, 4, 128, 64, 256
    A = torch.randn(b, n_kv, m, k, device='cuda', dtype=torch.float16)
    B = torch.randn(b, n_kv, k, n, device='cuda', dtype=torch.float16)
    
    # Warm up first
    _ = efficient_4d_matmul(A, B)
    torch.cuda.synchronize()

    start_time = time.time()
    C = efficient_4d_matmul(A, B)
    print(f"Triton kernel time: {time.time() - start_time}")

    start_time = time.time()
    C_ref = torch.matmul(A, B)
    print(f"Pytorch kernel time: {time.time() - start_time}")

    print("Max error:", (C.float() - C_ref.float()).abs().max())

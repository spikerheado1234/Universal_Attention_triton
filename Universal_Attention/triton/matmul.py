import triton
import triton.language as tl
import torch
import time

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
    assert A.ndim == 4 and B.ndim == 4, "Inputs must be 4-D"
    assert A.shape[-1] == B.shape[-2], "Dimension mismatch"

    b, n_kv, m, k = A.shape
    _, _, _, n = B.shape

    if A.stride(-1) != 1:
        A = A.contiguous()
    if A.device != B.device:
        B = B.to(A.device)
    if B.stride(-1) != 1:
        B = B.contiguous()

    C = torch.empty((b, n_kv, m, n), device=A.device, dtype=A.dtype)

    grid = lambda META: (b * n_kv, triton.cdiv(m, META['BLOCK_M']), triton.cdiv(n, META['BLOCK_N']))

    matmul_4d_kernel[grid](
        A, B, C,
        b, n_kv, m, k, n, 
        A.stride(0), A.stride(1), A.stride(2), A.stride(3),
        B.stride(0), B.stride(1), B.stride(2), B.stride(3),
        C.stride(0), C.stride(1), C.stride(2), C.stride(3),
    )
        
    return C

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
    key=['i', 'k', 'j'],
)
@triton.jit
def matmul_4d_kernel(
    A_ptr, B_ptr, C_ptr,
    b, n_kv, i, k, j, rep,
    stride_ab, stride_an_kv, stride_ai, stride_ak,
    stride_bb, stride_bn, stride_bj, stride_bk, 
    stride_cb, stride_cn, stride_ci, stride_cj,
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
    B_ptr += pid_b * stride_bb + pid_n_kv * rep * stride_bn
    C_ptr += pid_b * stride_cb + pid_n_kv * rep * stride_cn

    for r in range(0, rep):
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_offset in range(0, k, BLOCK_K):
            offs_k = k_offset + tl.arange(0, BLOCK_K)
            A_mat = tl.load(
                A_ptr + offs_m[:, None] * stride_ai + offs_k[None, :] * stride_ak,
                mask=(offs_m[:, None] < i) & (offs_k[None, :] < k),
                other=0.0,
            )
            B_mat = tl.load(
                B_ptr + r * stride_bn + offs_n[:, None] * stride_bj + offs_k[None, :] * stride_bk,
                mask=(offs_n[:, None] < j) & (offs_k[None, :] < k),
                other=0.0,
            )
            acc += tl.dot(A_mat, tl.trans(B_mat), input_precision="ieee")

        tl.store(
            C_ptr + r * stride_cn + offs_m[:, None] * stride_ci + offs_n[None, :] * stride_cj,
            acc,
            mask=(offs_m[:, None] < i) & (offs_n[None, :] < j),
        )

    # 3D mat mul is supported by triton, but not by llvm, so it is not used here. 

    # offs_rep = tl.arange(0, REP)
    # acc = tl.zeros((REP, BLOCK_M, BLOCK_N), dtype=tl.float32)

    # for k_offset in range(0, k, BLOCK_K):
    #     offs_k = k_offset + tl.arange(0, BLOCK_K)
    #     A_mat = tl.load(
    #         A_ptr + offs_m[:, None] * stride_ai + offs_k[None, :] * stride_ak,
    #         mask=(offs_m[:, None] < i) & (offs_k[None, :] < k),
    #         other=0.0,
    #     )
    #     A_mat = tl.broadcast_to(A_mat[None, :, :], (REP, BLOCK_M, BLOCK_K))

    #     B_mat = tl.load(
    #         B_ptr + 
    #         offs_rep[:, None, None] * stride_bn + 
    #         offs_n[None, :, None] * stride_bj + 
    #         offs_k[None, None, :] * stride_bk,
    #         mask=
    #         (offs_rep[:, None, None] < REP) &
    #         (offs_n[None, :, None] < j) & 
    #         (offs_k[None, None, :] < k),
    #         other=0.0,
    #     )
    #     acc += tl.dot(A_mat, tl.permute(B_mat, (0, 2, 1)))

    # tl.store(
    #     C_ptr + 
    #     offs_rep[:, None, None] * stride_cn + 
    #     offs_m[None, :, None] * stride_ci + 
    #     offs_n[None, None, :] * stride_cj,
    #     acc,
    #     mask=
    #     (offs_rep[:, None, None] < REP) &
    #     (offs_m[None, :, None] < i) & 
    #     (offs_n[None, None, :] < j),
    # )

def efficient_4d_matmul_rep(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    # assert A.ndim == 4 and B.ndim == 4, "Inputs must be 4-D"
    # assert A.shape[-1] == B.shape[-2], "Dimension mismatch"
    b, n_kv, i, k = A.shape
    _, n, j, _ = B.shape
    rep = n // n_kv

    if A.stride(-1) != 1:
        A = A.contiguous()
    if A.device != B.device:
        B = B.to(A.device)
    if B.stride(-1) != 1:
        B = B.contiguous()

    C = torch.empty((b, n_kv * rep, i, j), device=A.device, dtype=A.dtype)

    grid = lambda META: (b * n_kv, triton.cdiv(i, META['BLOCK_M']), triton.cdiv(j, META['BLOCK_N']))

    matmul_4d_kernel[grid](
        A, B, C,
        b, n_kv, i, k, j, rep,
        A.stride(0), A.stride(1), A.stride(2), A.stride(3), 
        B.stride(0), B.stride(1), B.stride(2), B.stride(3),
        C.stride(0), C.stride(1), C.stride(2), C.stride(3),
    )
        
    return C

if __name__ == "__main__":
    b, n_kv, i, k, j = 2, 4, 128, 64, 256
    rep = 4
    A = torch.randn(b, n_kv, i, k, device='cuda', dtype=torch.float32) * 10
    B = torch.randn(b, n_kv * rep, j, k, device='cuda', dtype=torch.float32) * 10
    
    # Warm up first
    _ = efficient_4d_matmul_rep(A, B)
    torch.cuda.synchronize()

    start_time = time.time()
    E = efficient_4d_matmul_rep(A, B)
    print(f"Triton kernel time: {time.time() - start_time}")

    start_time = time.time()
    C = A.view(b, n_kv, 1, i, k)
    D = B.view(b, n_kv, rep, j, k)
    E_ref = (C @ D.transpose(-1, -2)).view(b, n_kv * rep, i, j)
    # C_ref = torch.matmul(A, B)
    print(f"Pytorch kernel time: {time.time() - start_time}")

    print("Max error:", (E.float() - E_ref.float()).abs().max())

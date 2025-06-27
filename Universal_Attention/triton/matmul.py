import triton
import triton.language as tl
import torch
import time

from packaging import version
TRITON_32 = version.parse(triton.__version__) >= version.parse('3.2.0')

configs = [
    triton.Config({'BLOCK_K': BLOCK_K}, num_stages=stages, num_warps=warps) \
    for BLOCK_K in [32, 64, 128]\
    for stages in [2, 3, 4]\
    for warps in [2, 4, 8]\
]

@triton.autotune(
    configs=configs,
    key=['m', 'k', 'n'],
)
@triton.jit
def batched_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    b, n_kv, m, k, n, 
    stride_a_b, stride_a_n_kv, stride_a_m, stride_a_k,
    stride_b_b, stride_b_n_kv, stride_b_k, stride_b_n,
    stride_c_b, stride_c_n_kv, stride_c_m, stride_c_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
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
    C_ptr += pid_b * stride_c_b + pid_n_kv * stride_c_n_kv

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_offset in range(0, k, BLOCK_K):
        offs_k = k_offset + tl.arange(0, BLOCK_K)
        A_mat = tl.load(
            A_ptr + offs_m[:, None] * stride_a_m + offs_k[None, :] * stride_a_k,
            mask=(offs_m[:, None] < m) & (offs_k[None, :] < k),
            other=0.0,
        )
        B_mat = tl.load(
            B_ptr + offs_k[:, None] * stride_b_k + offs_n[None, :] * stride_b_n,
            mask=(offs_k[:, None] < k) & (offs_n[None, :] < n),
            other=0.0,
        )
        acc += tl.dot(A_mat, B_mat)

    tl.store(
        C_ptr + offs_m[:, None] * stride_c_m + offs_n[None, :] * stride_c_n,
        acc,
        mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
    )

def batched_matmul_triton(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.ndim == 4 and B.ndim == 4, "Inputs must be 4-D"
    assert A.shape[-1] == B.shape[-2], "Dimension mismatch"

    b, n_kv, m, k = A.shape
    _, _, _, n = B.shape

    BLOCK_M = 64
    BLOCK_N = 64
    M_BLOCK = triton.cdiv(m, BLOCK_M)
    N_BLOCK = triton.cdiv(n, BLOCK_N)

    if A.stride(-1) != 1:
        A = A.contiguous()
    if A.device != B.device:
        B = B.to(A.device)
    if B.stride(-1) != 1:
        B = B.contiguous()

    C = torch.empty((b, n_kv, m, n), device=A.device, dtype=A.dtype)

    # grid = lambda META: (b * n_kv, triton.cdiv(m, META['BLOCK_M']), triton.cdiv(n, META['BLOCK_N']))
    grid = (b * n_kv, M_BLOCK, N_BLOCK)

    broadcasted_matmul_kernel[grid](
        A, B, C,
        b, n_kv, m, k, n, 
        A.stride(0), A.stride(1), A.stride(2), A.stride(3),
        B.stride(0), B.stride(1), B.stride(2), B.stride(3),
        C.stride(0), C.stride(1), C.stride(2), C.stride(3),
        BLOCK_M, BLOCK_N, 
    )
        
    return C

@triton.autotune(
    configs=configs,
    key=['m', 'k', 'n'],
)
@triton.jit
def broadcasted_2D_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    b, n_kv, rep, m, k, n, 
    stride_a_b, stride_a_n_kv, stride_a_m, stride_a_k,
    stride_b_b, stride_b_n_rep, stride_b_n, stride_b_k, 
    stride_c_b, stride_c_n_rep, stride_c_m, stride_c_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_b_n_kv = tl.program_id(0)
    pid_b = pid_b_n_kv // n_kv
    pid_n_kv = pid_b_n_kv % n_kv
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    A_ptr += pid_b * stride_a_b + pid_n_kv * stride_a_n_kv
    B_ptr += pid_b * stride_b_b + pid_n_kv * rep * stride_b_n_rep
    C_ptr += pid_b * stride_c_b + pid_n_kv * rep * stride_c_n_rep

    for r in range(0, rep):
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_offset in range(0, k, BLOCK_K):
            offs_k = k_offset + tl.arange(0, BLOCK_K)
            A_mat = tl.load(
                A_ptr + offs_m[:, None] * stride_a_m + offs_k[None, :] * stride_a_k,
                mask=(offs_m[:, None] < m) & (offs_k[None, :] < k),
                other=0.0,
            )
            B_mat = tl.load(
                B_ptr + r * stride_b_n_rep + offs_n[:, None] * stride_b_n + offs_k[None, :] * stride_b_k,
                mask=(offs_n[:, None] < n) & (offs_k[None, :] < k),
                other=0.0,
            )
            acc += tl.dot(A_mat, tl.trans(B_mat), input_precision="ieee")

        tl.store(
            C_ptr + r * stride_c_n_rep + offs_m[:, None] * stride_c_m + offs_n[None, :] * stride_c_n,
            acc,
            mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
        )

@triton.autotune(
    configs=configs,
    key=['m', 'k', 'n'],
)
@triton.jit
def broadcasted_3D_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    b, n_kv, m, k, n, 
    stride_a_b, stride_a_n_kv, stride_a_m, stride_a_k,
    stride_b_b, stride_b_n_rep, stride_b_n, stride_b_k, 
    stride_c_b, stride_c_n_rep, stride_c_m, stride_c_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, REP: tl.constexpr,
):
    pid_b_n_kv = tl.program_id(0)
    pid_b = pid_b_n_kv // n_kv
    pid_n_kv = pid_b_n_kv % n_kv
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    A_ptr += pid_b * stride_a_b + pid_n_kv * stride_a_n_kv
    B_ptr += pid_b * stride_b_b + pid_n_kv * REP * stride_b_n_rep
    C_ptr += pid_b * stride_c_b + pid_n_kv * REP * stride_c_n_rep

    offs_rep = tl.arange(0, REP)
    acc = tl.zeros((REP, BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_offset in range(0, k, BLOCK_K):
        offs_k = k_offset + tl.arange(0, BLOCK_K)
        A_mat = tl.load(
            A_ptr + offs_m[:, None] * stride_a_m + offs_k[None, :] * stride_a_k,
            mask=(offs_m[:, None] < m) & (offs_k[None, :] < k),
            other=0.0,
        )
        A_mat = tl.broadcast_to(A_mat[None, :, :], (REP, BLOCK_M, BLOCK_K))

        B_mat = tl.load(
            B_ptr + 
            offs_rep[:, None, None] * stride_b_n_rep + 
            offs_n[None, :, None] * stride_b_n + 
            offs_k[None, None, :] * stride_b_k,
            mask=
            (offs_rep[:, None, None] < REP) &
            (offs_n[None, :, None] < n) & 
            (offs_k[None, None, :] < k),
            other=0.0,
        )
        acc += tl.dot(A_mat, tl.permute(B_mat, (0, 2, 1)))

    # Storing after 3D matmul is fixed in 3.2.0
    tl.store(
        C_ptr + 
        offs_rep[:, None, None] * stride_c_n_rep + 
        offs_m[None, :, None] * stride_c_m + 
        offs_n[None, None, :] * stride_c_n,
        acc,
        mask=
        (offs_rep[:, None, None] < REP) &
        (offs_m[None, :, None] < m) & 
        (offs_n[None, None, :] < n),
    )

def broadcasted_matmul_triton(A: torch.Tensor, B: torch.Tensor, _3D=False) -> torch.Tensor:
    b, n_kv, m, k = A.shape
    _, n_rep, n, _ = B.shape
    rep = n_rep // n_kv

    BLOCK_M = 64
    BLOCK_N = 64
    M_BLOCK = triton.cdiv(m, BLOCK_M)
    N_BLOCK = triton.cdiv(n, BLOCK_N)

    if A.stride(-1) != 1:
        A = A.contiguous()
    if A.device != B.device:
        B = B.to(A.device)
    if B.stride(-1) != 1:
        B = B.contiguous()

    C = torch.empty((b, n_kv * rep, m, n), device=A.device, dtype=A.dtype)

    # grid = lambda META: (b * n_kv, triton.cdiv(m, META['BLOCK_M']), triton.cdiv(n, META['BLOCK_N']))
    grid = (b * n_kv, M_BLOCK, N_BLOCK)

    if _3D and TRITON_32:
        broadcasted_3D_matmul_kernel[grid](
            A, B, C,
            b, n_kv, m, k, n, 
            A.stride(0), A.stride(1), A.stride(2), A.stride(3), 
            B.stride(0), B.stride(1), B.stride(2), B.stride(3),
            C.stride(0), C.stride(1), C.stride(2), C.stride(3),
            BLOCK_M, BLOCK_N, REP=rep,
        )
    else:
        broadcasted_2D_matmul_kernel[grid](
            A, B, C,
            b, n_kv, rep, m, k, n, 
            A.stride(0), A.stride(1), A.stride(2), A.stride(3), 
            B.stride(0), B.stride(1), B.stride(2), B.stride(3),
            C.stride(0), C.stride(1), C.stride(2), C.stride(3),
            BLOCK_M, BLOCK_N, 
        )
        
    return C

if __name__ == "__main__":
    b, n_kv, m, k, n = 2, 4, 1024, 512, 2048
    rep = 4
    _3D = True
    A = torch.randn(b, n_kv, m, k, device='cuda', dtype=torch.float32) * 10
    B = torch.randn(b, n_kv * rep, n, k, device='cuda', dtype=torch.float32) * 10
    
    # Warm up first
    _ = broadcasted_matmul_triton(A, B, _3D=_3D)
    torch.cuda.synchronize()

    start_time = time.time()
    C = broadcasted_matmul_triton(A, B, _3D=_3D)
    print(f"Triton kernel time: {time.time() - start_time}")

    start_time = time.time()
    A_perm = A.view(b, n_kv, 1, m, k)
    B_perm = B.view(b, n_kv, rep, n, k)
    C_ref = (A_perm @ B_perm.transpose(-1, -2)).view(b, n_kv * rep, m, n)
    # C_ref = torch.matmul(A, B)
    print(f"Pytorch kernel time: {time.time() - start_time}")

    print("Max error:", (C.float() - C_ref.float()).abs().max())

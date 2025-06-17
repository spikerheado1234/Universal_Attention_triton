import triton
import triton.language as tl
import torch

@triton.jit
def matmul_4d_kernel(
    A_ptr, B_ptr, C_ptr,
    BATCH, S, D,
    stride_ab, stride_as, stride_ad,
    stride_bb, stride_bd, stride_bs,
    stride_cb, stride_cs, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # program axes
    pid_batch = tl.program_id(0)
    pid_m     = tl.program_id(1)
    pid_n     = tl.program_id(2)

    # compute row/col offsets for this tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # rows in [0..S)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # cols in [0..S)

    # pointers for this (b, n_kv) pair
    A_batch_ptr = A_ptr + pid_batch * stride_ab
    B_batch_ptr = B_ptr + pid_batch * stride_bb
    C_batch_ptr = C_ptr + pid_batch * stride_cb

    # accumulator register block
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # loop over K dimension in tiles
    for k_offset in range(0, D, BLOCK_K):
        offs_k = k_offset + tl.arange(0, BLOCK_K)
        # load A[batch, offs_m, offs_k] -> [BLOCK_M, BLOCK_K]
        a = tl.load(
            A_batch_ptr
            + offs_m[:, None] * stride_as
            + offs_k[None, :] * stride_ad,
            mask=(offs_m[:, None] < S) & (offs_k[None, :] < D),
            other=0.0,
        )
        # load B[batch, offs_k, offs_n] -> [BLOCK_K, BLOCK_N]
        b = tl.load(
            B_batch_ptr
            + offs_k[:, None] * stride_bd
            + offs_n[None, :] * stride_bs,
            mask=(offs_k[:, None] < D) & (offs_n[None, :] < S),
            other=0.0,
        )
        # accumulate
        acc += tl.dot(a, b)

    # store C[batch, offs_m, offs_n]
    tl.store(
        C_batch_ptr
        + offs_m[:, None] * stride_cs
        + offs_n[None, :] * stride_cn,
        acc,
        mask=(offs_m[:, None] < S) & (offs_n[None, :] < S),
    )


def efficient_4d_matmul(A: torch.Tensor, B: torch.Tensor, BLOCK_S: int = 32, BLOCK_K: int = 16) -> torch.Tensor:
    """
    Perform A@B where:
      A: [b, n_kv, S, D]
      B: [b, n_kv, D, S]
    returning C: [b, n_kv, S, S]
    using tiled Triton kernel to chunk over S and D.
    """
    assert A.ndim == 4 and B.ndim == 4, "Inputs must be 4-D"
    b, n_kv, S, D = A.shape
    C = torch.empty((b, n_kv, S, S), device=A.device, dtype=A.dtype)

    # flatten batch & n_kv dims
    BATCH = b * n_kv
    A_flat = A.contiguous().view(BATCH, S, D)
    B_flat = B.contiguous().view(BATCH, D, S)
    C_flat = C.contiguous().view(BATCH, S, S)

    # pointers and strides
    A_ptr = A_flat.data_ptr()
    B_ptr = B_flat.data_ptr()
    C_ptr = C_flat.data_ptr()
    stride_ab, stride_as, stride_ad = A_flat.stride(0), A_flat.stride(1), A_flat.stride(2)
    stride_bb, stride_bd, stride_bs = B_flat.stride(0), B_flat.stride(1), B_flat.stride(2)
    stride_cb, stride_cs, stride_cn = C_flat.stride(0), C_flat.stride(1), C_flat.stride(2)

    # grid dims
    grid = (
        BATCH,
        (S + BLOCK_S - 1) // BLOCK_S,
        (S + BLOCK_S - 1) // BLOCK_S,
    )

    matmul_4d_kernel[grid](
        A_ptr, B_ptr, C_ptr,
        BATCH, S, D,
        stride_ab, stride_as, stride_ad,
        stride_bb, stride_bd, stride_bs,
        stride_cb, stride_cs, stride_cn,
        BLOCK_M=BLOCK_S, BLOCK_N=BLOCK_S, BLOCK_K=BLOCK_K,
    )
    return C


# Example usage:
if __name__ == "__main__":
    # random test
    b, n_kv, S, D = 2, 4, 128, 32
    A = torch.randn(b, n_kv, S, D, device='cuda', dtype=torch.float16)
    B = torch.randn(b, n_kv, D, S, device='cuda', dtype=torch.float16)
    C = efficient_4d_matmul(A, B, BLOCK_S=32, BLOCK_K=16)
    # verify against torch
    C_ref = torch.matmul(A, B)
    print("Max error:", (C.float() - C_ref.float()).abs().max())

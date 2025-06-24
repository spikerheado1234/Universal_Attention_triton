import torch
import triton
import triton.language as tl


# class UniversalAttention(Function):
#     @staticmethod
#     def forward(q, k, v, src, dest):
#         b, n, s, d = q.shape
#         _, n_kv, _, _ = k.shape
#           q.shape == (b, n, s, d)
#         assert k.shape == (b, n_kv, s, d)
#         assert v.shape == (b, n_kv, s, d)
#         assert src.shape == (b, n_kv, s)
#         assert dest.shape == (b, n_kv, s)
#         if q.stride(-1) != 1:
#             q = q.contiguous()
#         if k.stride(-1) != 1:
#             k = k.contiguous()
#         if v.stride(-1) != 1:
#             v = v.contiguous()
#         if src.stride(-1) != 1:
#             src = src.contiguous()
#         if dest.stride(-1) != 1:
#             dest = dest.contiguous()
#         output = _universal_attention_fwd(q, k, v, src, dest)
#         return output

#     @staticmethod
#     def setup_context(ctx, inputs, outputs):
#         q, k, v, src, dest = inputs
#         # out, denom = outputs
#         ctx.save_for_backward(q, k, v, src, dest)

#     @staticmethod
#     def backward(ctx, doutput):
#         # Note: when using mixed precision, dout is downcast but ddenom is always fp32
#         q, k, v, src, dest = ctx.saved_tensors
#         b, n, s, d = q.shape
#         _, n_kv, _, _ = k.shape
#         assert doutput.shape == (b, n, s, d)
#         if doutput.stride(-1) != 1:
#             doutput = doutput.contiguous()
#         dq, dk, dv, dsrc, ddest = _universal_attention_bwd(q, k, v, src, dest, doutput)
#         return dq, dk, dv, dsrc, ddest


'''
######################################
#     Forward Kernel & Interface     #
######################################
'''
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 128, 'BLOCK_D': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_C': 64, 'BLOCK_D': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_C': 128, 'BLOCK_D': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_C': 128, 'BLOCK_D': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_C': 64, 'BLOCK_D': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_C': 128, 'BLOCK_D': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_C': 64, 'BLOCK_D': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_C': 128, 'BLOCK_D': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_C': 64, 'BLOCK_D': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_C': 32, 'BLOCK_D': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_C': 64, 'BLOCK_D': 32}, num_stages=4, num_warps=2),
    ],
    key=['s', 'd'],
)
@triton.jit
def _universal_attention_fwd_kernel(
    # Pointers to matrices
    q_ptr, k_ptr, v_ptr, src_ptr, dest_ptr, output_ptr, semaphore_ptr,
    # Matrix dimensions
    b, n_kv, rep, s, d, 
    # Strides
    stride_q_b, stride_q_n, stride_q_s, stride_q_d, 
    stride_k_b, stride_k_n_kv, stride_k_s, stride_k_d, 
    stride_v_b, stride_v_n_kv, stride_v_s, stride_v_d,
    stride_src_b, stride_src_n_kv, stride_src_s, 
    stride_dest_b, stride_dest_n_kv, stride_dest_s, 
    stride_output_b, stride_output_n, stride_output_s, stride_output_d, 
    # Meta-parameters
    BLOCK_C: tl.constexpr, BLOCK_D: tl.constexpr, 
):
    pid_b_n_kv = tl.program_id(0)
    pid_b = pid_b_n_kv // n_kv
    pid_n_kv = pid_b_n_kv % n_kv
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    offs_m = pid_m * BLOCK_C + tl.arange(0, BLOCK_C)
    offs_n = pid_n * BLOCK_C + tl.arange(0, BLOCK_C)

    # k @ k^T
    k_ptr += pid_b * stride_k_b + pid_n_kv * stride_k_n_kv
    acc = tl.zeros((BLOCK_C, BLOCK_C), dtype=tl.float32)

    for d_offset in range(0, d, BLOCK_D):
        offs_d = d_offset + tl.arange(0, BLOCK_D)
        k_mat = tl.load(
            k_ptr + offs_m[:, None] * stride_k_s + offs_d[None, :] * stride_k_d,
            mask=offs_m[:, None] < s & offs_d[None, :] < d,
            other=0.0
        )
        kt_mat = tl.load(
            k_ptr + offs_n[:, None] * stride_k_s + offs_d[None, :] * stride_k_d,
            mask=offs_n[:, None] < s & offs_d[None, :] < d,
            other=0.0
        )
        acc += tl.dot(k_mat, kt_mat)
    
    # .relu()
    acc = tl.maximum(acc, 0.0)

    # .pow(2/3)
    acc = tl.exp2(tl.log2(acc) * 2.0 / 3.0)

    # * static_src_.pow(1/3).unsqueeze(-1) * static_dest.pow(1/3).unsqueeze(-2)
    src_ptr += pid_b * stride_src_b + pid_n_kv * stride_src_n_kv
    dest_ptr += pid_b * stride_dest_b + pid_n_kv * stride_dest_n_kv

    src_mat  = tl.load(
        src_ptr  + offs_m * stride_src_s,
        mask=offs_m < s, 
        other=0.0
    )
    src_mat = tl.exp2(tl.log2(src_mat) / 3.0)
    
    dest_mat = tl.load(dest_ptr + offs_n * stride_dest_s,
        mask=offs_n < s, 
        other=0.0
    )
    dest_mat = tl.exp2(tl.log2(dest_mat) / 3.0)

    acc = acc * src_mat[:, None] * dest_mat[None, :]

    tl.store(
        C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        acc,
        mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
    )

def _universal_attention_fwd(q, k, v, src, dest):
    b, n, s, d = q.shape
    _, n_kv, _, _ = k.shape
    assert n % n_kv == 0, "n needs to be divisible by n_kv"
    rep = n // n_kv
    device = q.device

    # chunk sequence length: META['BLOCK_C']
    
    output = torch.empty(b, n, s, d, dtype=q.dtype, device=device)
    semaphore = torch.zeros(b, n_kv, s, dtype=torch.int32, device=device)

    grid = lambda META: (b * n_kv, triton.cdiv(s, META['BLOCK_C']), triton.cdiv(s, META['BLOCK_C']))

    _universal_attention_fwd_kernel[grid](
        q, k, v, src, dest, output, semaphore,
        b, n_kv, rep, s, d, 
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),                         # (b, n, s, d)
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),                         # (b, n_kv, s, d)
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),                         # (b, n_kv, s, d)
        src.stride(0), src.stride(1), src.stride(2),                                # (b, n_kv, s)
        dest.stride(0), dest.stride(1), dest.stride(2),                             # (b, n_kv, s)
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),     # (b, n, s, d)
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
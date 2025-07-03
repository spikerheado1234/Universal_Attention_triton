import torch
import triton
import triton.language as tl

import numpy as np
import inspect

configs = [
    triton.Config({'BLOCK_D': BLOCK_D}, num_stages=stages, num_warps=warps) \
    for BLOCK_D in [32, 64, 128]\
    for stages in [2, 3, 4]\
    for warps in [2, 4, 8]\
]

'''
######################################
#     Forward Kernel & Interface     #
######################################
'''
@triton.autotune(
    configs=configs,
    key=['r', 'n_', '_n', 'd'],
)
@triton.jit
def _universal_attention_fwd_kernel(
    # Pointers to matrices
    kc, vc, xq, kt, src, dest, out, denom, buffer,                               
    # Matrix dimensions
    b, h, r, n_, _n, d, 
    # Strides
    str_kc_b, str_kc_h, str_kc_n_, str_kc_c_, str_kc_d,                     # b h n_ c_ d
    str_vc_b, str_vc_h, str_vc_n_, str_vc_c_, str_vc_d,                     # b h n_ c_ d
    str_xq_b, str_xq_h, str_xq_r, str_xq__n, str_xq__c, str_xq_d,           # b h r _n _c d
    str_kt_b, str_kt_h, str_kt_d, str_kt__n, str_kt__c,                     # b h d _n _c
    str_src_b, str_src_h, str_src_n_, str_src_c_,                           # b h n_ c_
    str_dest_b, str_dest_h, str_dest__n, str_dest__c,                       # b h _n _c
    str_out_b, str_out_h, str_out_r, str_out_l, str_out_d, str_out_n_,      # b h r l d n_
    str_denom_b, str_denom_h, str_denom_r, str_denom_l, str_denom_n_,       # b h r l n_
    str_buffer_b, str_buffer_h, str_buffer_n_, str_buffer_c_,               # b h n_ c_
    # Meta-parameters
    BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr, BLOCK_D: tl.constexpr,    # Block dims
    IDX_J: tl.constexpr, DTYPE: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_i = tl.program_id(2)                # n_
    pid_j = IDX_J                           # _n  

    offs_i = tl.arange(0, BLOCK_C)          # c_
    offs_j = tl.arange(0, BLOCK_R)          # _c

    offs_tri = pid_i * BLOCK_C - pid_j * BLOCK_R
    offs_block = pid_j * BLOCK_R + offs_j

    affinity = tl.zeros((BLOCK_C, BLOCK_R), dtype=tl.float32)

    # k_.matmul(_kt)
    kc_ptr = kc + pid_b * str_kc_b + pid_h * str_kc_h + pid_i * str_kc_n_
    kt_ptr = kt + pid_b * str_kt_b + pid_h * str_kt_h + pid_j * str_kt__n

    for d_offset in range(0, d, BLOCK_D):
        offs_d = d_offset + tl.arange(0, BLOCK_D)
        kc_mat = tl.load(
            kc_ptr + offs_i[:, None] * str_kc_c_ + offs_d[None, :] * str_kc_d, 
            mask=(offs_i[:, None] < BLOCK_C) & (offs_d[None, :] < d), 
            other=0.0
        )
        kc_mat = tl.cast(kc_mat, tl.float32)

        kt_mat = tl.load(
            kt_ptr + offs_d[:, None] * str_kt_d + offs_j[None, :] * str_kt__c, 
            mask=(offs_d[:, None] < d) & (offs_j[None, :] < BLOCK_R), 
            other=0.0
        )
        kt_mat = tl.cast(kt_mat, tl.float32)

        # Use ieee to use fp32, otherwise the default would be tf32 even after tl.cast
        affinity += tl.dot(kc_mat, kt_mat, input_precision="ieee")

    # .relu()
    affinity = tl.maximum(affinity, 0.0)

    # .pow(2/3)
    affinity = tl.exp2(tl.log2(affinity) * 2.0 / 3.0)

    # * static_src_.pow(1/3).unsqueeze(-1) * static_dest.pow(1/3).unsqueeze(-2)
    src_ptr = src + pid_b * str_src_b + pid_h * str_src_h + pid_i * str_src_n_ 
    src_mat = tl.load(src_ptr + offs_i * str_src_c_, mask=offs_i < BLOCK_C, other=0.0)
    src_mat = tl.cast(src_mat, tl.float32)
    src_mat = tl.exp2(tl.log2(src_mat) / 3.0)

    dest_ptr = dest + pid_b * str_dest_b + pid_h * str_dest_h + pid_j * str_dest__n
    dest_mat = tl.load(dest_ptr + offs_j * str_dest__c, mask=offs_j < BLOCK_R, other=0.0)
    dest_mat = tl.cast(dest_mat, tl.float32)
    dest_mat = tl.exp2(tl.log2(dest_mat) / 3.0)

    affinity = affinity * src_mat[:, None] * dest_mat[None, :]

    # torch.log1p(affinity.clamp(min=0, max=1-1e-6).neg())
    affinity = tl.clamp(affinity, 0.0, 1.0 - 1e-6)
    affinity = tl.log(1.0 - affinity) 

    # .triu(i*c_-j*_c+1)
    affinity = tl.where((offs_j[None, :] > (offs_i[:, None] + offs_tri)), affinity, 0.0)

    # Update sum buffer 
    curr_sum = tl.sum(affinity, axis=1, keep_dims=False) 
    buffer_ptr = buffer + pid_b * str_buffer_b + pid_h * str_buffer_h + pid_i * str_buffer_n_ 
    if pid_j == 0:
        # Initialize buffer to 0s
        prev_sum = tl.zeros((BLOCK_C,), dtype=tl.float32)
    else:
        prev_sum = tl.load(buffer_ptr + offs_i * str_buffer_c_, mask=offs_i < BLOCK_C, other=0.0) 
    tl.store(buffer_ptr + offs_i * str_buffer_c_, curr_sum + prev_sum, mask=offs_i < BLOCK_C)

    # .cumsum(3)
    affinity = tl.cumsum(affinity, axis=1) + prev_sum[:, None]  

    # .masked_fill(mask.tril(i*c_-j*_c-1), -1e12)
    affinity = tl.where((offs_j[None, :] < (offs_i[:, None] + offs_tri)), -1e12, affinity)
    affinity = tl.cast(affinity, tl.float32)

    # k_.unsqueeze(2).matmul(_q.transpose(-1,-2)).add(affinity.unsqueeze(2))
    vc_ptr = vc + pid_b * str_vc_b + pid_h * str_vc_h + pid_i * str_vc_n_
    xq_ptr = xq + pid_b * str_xq_b + pid_h * str_xq_h + pid_j * str_xq__n

    denom_ptr = denom + pid_b * str_denom_b + pid_h * str_denom_h + pid_i * str_denom_n_ 
    denom_ptr = denom_ptr + offs_block * str_denom_l

    out_ptr = out + pid_b * str_out_b + pid_h * str_out_h + pid_i * str_out_n_
    out_ptr = out_ptr + offs_block * str_out_l

    # @Haochen: storing a 3D tensor after applying .dot() to 3D tensors would fail LLVM compilation
    # The problem is fixed in triton 3.2.0, and the alternative code is listed in matmul.py
    for rep in range(0, r):
        xq_rep_ptr = xq_ptr + rep * str_xq_r
        denom_rep_ptr = denom_ptr + rep * str_denom_r
        out_rep_ptr = out_ptr + rep * str_out_r

        kq = tl.zeros((BLOCK_C, BLOCK_R), dtype=tl.float32)
        for d_offset in range(0, d, BLOCK_D):
            offs_d = d_offset + tl.arange(0, BLOCK_D)
            kc_mat = tl.load(
                kc_ptr + offs_i[:, None] * str_kc_c_ + offs_d[None, :] * str_kc_d, 
                mask=(offs_i[:, None] < BLOCK_C) & (offs_d[None, :] < d), 
                other=0.0
            )
            kc_mat = tl.cast(kc_mat, tl.float32)

            xq_mat = tl.load(
                xq_rep_ptr + offs_j[:, None] * str_xq__c + offs_d[None, :] * str_xq_d, 
                mask=(offs_j[:, None] < BLOCK_R) & (offs_d[None, :] < d), 
                other=0.0
            )
            xq_mat = tl.cast(xq_mat, tl.float32)

            # Use ieee to use fp32, otherwise the default would be tf32 even after tl.cast
            kq += tl.dot(kc_mat,tl.trans(xq_mat), input_precision="ieee")

        score = kq + affinity

        # Stabilize logsumexp using the subtract max trick
        score_max = tl.max(score, axis=0) 
        score_shifted = score - score_max[None, :]
        score_exp = tl.exp(score_shifted)
        score_sumexp = tl.sum(score_exp, axis=0)
        score_logsumexp = score_max + tl.log(score_sumexp)

        tl.store(denom_rep_ptr, score_logsumexp, mask=offs_block < BLOCK_R * _n)

        # score.transpose(-1,-2).softmax(dim=-1).to(dtype=_q.dtype).matmul(v_.unsqueeze(2))
        score_softmax = tl.div_rn(tl.trans(score_exp), score_sumexp[:, None])
        score_softmax = tl.cast(score_softmax, DTYPE) 
        
        for d_offset in range(0, d, BLOCK_D):
            offs_d = d_offset + tl.arange(0, BLOCK_D)
            vc_mat = tl.load(
                vc_ptr + offs_i[:, None] * str_vc_c_ + offs_d[None, :] * str_vc_d, 
                mask=(offs_i[:, None] < BLOCK_C) & (offs_d[None, :] < d), 
                other=0.0
            )
            # vc_mat = tl.cast(vc_mat, tl.float32)
            softmax_v = tl.dot(score_softmax, vc_mat, input_precision="ieee")

            tl.store(
                out_rep_ptr[:, None] + offs_d[None, :] * str_out_d, 
                softmax_v, 
                mask=(offs_block[:, None] < BLOCK_R * _n) & (offs_d[None, :] < d), 
            )

def _universal_attention_fwd(kc, vc, xq, static_src, static_dest):
    '''
    Inputs:
    kc: b h n_ c_ d
    vc: b h n_ c_ d
    xq: b h r _n _c d
    static_src: b h n_ c_
    static_dest: b h _n _c

    Outputs:
    out: b h r l d n_
    denom: b h r l n_

    Intermediate: 
    variables_ are split over cols
    _variables are split over rows
    (i.e. n_ is the number of col chunks, _n is the number of row chunks)
    '''    
    b,h,r,_n,_c,d = xq.shape
    _,_,n_,c_,_ = kc.shape
    l = n_*c_
    dtype = tl.float16 if xq.dtype == torch.float16 else tl.float32

    out = torch.empty(b,h,r,l,d,n_, dtype=xq.dtype, device=xq.device)
    denom = torch.empty(b,h,r,l,n_, dtype=xq.dtype, device=xq.device)
    sum_buffer = torch.empty(b,h,n_,c_, dtype=static_src.dtype, device=static_src.device)

    kt = kc.view(b,h,_n,_c,d).permute(0,1,4,2,3)  # b h d _n _c

    # Due to the sum buffer for cumulative sum, we want to process that dimension sequencially
    grid = (b,h,n_)

    for j in range(_n):
        _universal_attention_fwd_kernel[grid](
            kc, vc, xq, kt, static_src, static_dest, out, denom, sum_buffer,                                                 
            b, h, r, n_, _n, d, 
            kc.stride(0), kc.stride(1), kc.stride(2), kc.stride(3), kc.stride(4),   
            vc.stride(0), vc.stride(1), vc.stride(2), vc.stride(3), vc.stride(4),   
            xq.stride(0), xq.stride(1), xq.stride(2), xq.stride(3), xq.stride(4), xq.stride(5),  
            kt.stride(0), kt.stride(1), kt.stride(2), kt.stride(3), kt.stride(4),   
            static_src.stride(0), static_src.stride(1), static_src.stride(2), static_src.stride(3), 
            static_dest.stride(0), static_dest.stride(1), static_dest.stride(2), static_dest.stride(3), 
            out.stride(0), out.stride(1), out.stride(2), out.stride(3), out.stride(4), out.stride(5), 
            denom.stride(0), denom.stride(1), denom.stride(2), denom.stride(3), denom.stride(4), 
            sum_buffer.stride(0), sum_buffer.stride(1), sum_buffer.stride(2), sum_buffer.stride(3),
            BLOCK_R=_c, BLOCK_C=c_, IDX_J=j, DTYPE=dtype, 
        )
    return out, denom

'''
#################################
#     Ground Truth Autograd     #
#################################
'''
def universal_attention_forward(kc, vc, xq, static_src, static_dest):
    '''
    Inputs:
    kc: b h n_ c_ d
    vc: b h n_ c_ d
    xq: b h r _n _c d
    static_src: b h n_ c_
    static_dest: b h _n _c

    Outputs:
    out: b h r l d n_
    denom: b h r l n_

    Intermediate: 
    variables_ are split over cols
    _variables are split over rows
    (i.e. n_ is the number of col chunks, _n is the number of row chunks)
    '''    
    b,h,r,_n,_c,d = xq.shape
    _,_,n_,c_,_ = kc.shape
    l = n_ * c_
    mask = torch.ones(c_,_c, dtype=torch.bool, device=xq.device)
    out = torch.empty(b,h,r,l,d,n_, dtype=xq.dtype, device=xq.device)
    denom = torch.empty(b,h,r,l,n_, dtype=xq.dtype, device=xq.device)
    static_src = static_src.pow(1/3)
    static_dest = static_dest.pow(1/3)
    kt = kc.view(b,h,_n,_c,d).permute(0,1,4,2,3)  # b h d _n _c
    
    # Iteration over columns
    for i in range(n_):
        k_ = kc[:,:,i]  # b h c_ d
        v_ = vc[:,:,i]  # b h c_ d
        static_src_ = static_src[:,:,i]  # b h c_
        sum_buffer = torch.zeros(b,h,c_, dtype=static_src.dtype, device=static_src.device)

        # Iteration over rows
        for j in range(_n):
            _q = xq[:,:,:,j]  # b h r _c d
            _static_dest = static_dest[:,:,j]  # b h _c
            _kt = kt[:,:,:,j]  # b h d _c

            # Calculate decay matrix
            affinity = k_.matmul(_kt).relu().pow(2/3).float()  # deltanet style decay: b h c_ _c
            affinity = affinity * static_src_.unsqueeze(-1) * _static_dest.unsqueeze(-2)  # incorporate mamba-style and per-token decay
            affinity = torch.log1p(affinity.clamp(min=0, max=1-1e-6).neg())  # b h c_ _c
            affinity = affinity.triu(i*c_-j*_c+1).cumsum(3)  # Accumulate decay with causal masking, within row chunk
            affinity = affinity + sum_buffer.unsqueeze(-1)  # Accumulate decay from prior row chunk
            sum_buffer = affinity[:,:,:,-1]
            affinity = affinity.masked_fill(mask.tril(i*c_-j*_c-1), -1e12)  # Re-mask, with 1s on diagonal

            # Perform actual attention operation
            score = k_.unsqueeze(2).matmul(_q.transpose(-1,-2)).add(affinity.unsqueeze(2))  # b h r c_ _c
            _denom_ = score.logsumexp(dim=-2)  # b h r _c
            _out_ = score.transpose(-1,-2).softmax(dim=-1).to(dtype=_q.dtype).matmul(v_.unsqueeze(2))  # b h r _c d

            out[:,:,:,j*_c:(j+1)*_c,:,i] = _out_
            denom[:,:,:,j*_c:(j+1)*_c,i] = _denom_
    return out, denom


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

if __name__ == "__main__":
    # b, h, r, n_, c_, _n, _c, d = 1, 1, 1, 4, 16, 4, 16, 32
    b, h, r, n_, c_, _n, _c, d = 2, 4, 2, 32, 64, 32, 64, 512

    kc = torch.rand((b, h, n_, c_, d), device='cuda', dtype=torch.float32)
    vc = torch.rand((b, h, n_, c_, d), device='cuda', dtype=torch.float32)
    xq = torch.rand((b, h, r, _n, _c, d), device='cuda', dtype=torch.float32)
    static_src = torch.rand((b, h, n_, c_), device='cuda', dtype=torch.float32)
    static_dest = torch.rand((b, h, _n, _c), device='cuda', dtype=torch.float32)

    out, denom = _universal_attention_fwd(kc, vc, xq, static_src, static_dest)
    out_ref, denom_ref = universal_attention_forward(kc, vc, xq, static_src, static_dest)

    # Convert to final output for sanity check
    output = out.mul(denom.softmax(dim=-1).unsqueeze(-2)).sum(-1)
    output_ref = out_ref.mul(denom_ref.softmax(dim=-1).unsqueeze(-2)).sum(-1)

    # count = 0
    # for arr in (denom, denom_ref):
    #     arr = arr.cpu().numpy().reshape(-1, n_)
    #     np.savetxt(f"denom{count}.csv", arr, delimiter=",", fmt="%.6f")   
    #     count += 1

    # count = 0
    # for arr in (out, out_ref):
    #     arr = arr.cpu().numpy().reshape(-1, n_ * d)
    #     np.savetxt(f"out{count}.csv", arr, delimiter=",", fmt="%.6f")   
    #     count += 1

    # count = 0
    # for arr in (output, output_ref):
    #     arr = arr.cpu().numpy().reshape(-1, d)
    #     np.savetxt(f"output{count}.csv", arr, delimiter=",", fmt="%.6f")   
    #     count += 1

    print("Checking denom:")
    torch.testing.assert_close(denom, denom_ref, atol=1e-3, rtol=1e-5)
    print("Checking out:")
    torch.testing.assert_close(out, out_ref, atol=1e-3, rtol=1e-5)
    print("Checking output:")
    torch.testing.assert_close(output, output_ref, atol=1e-3, rtol=1e-5)



import torch
import triton
import triton.language as tl

import numpy as np
import inspect
import time
import pdb

configs = [
    triton.Config({'BLOCK_D': BLOCK_D}, num_stages=stages, num_warps=warps) \
    for BLOCK_D in [16, 32, 64]\
    for stages in [1, 2, 3]\
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
    kc, vc, xq, kt, src, dest, out, denom,                                
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
    # Meta-parameters
    BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr, BLOCK_D: tl.constexpr,    # Block dims
    DTYPE: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_i = tl.program_id(2)                # n_

    offs_i = tl.arange(0, BLOCK_C)          # c_
    offs_j = tl.arange(0, BLOCK_R)          # _c
    offs_k = tl.arange(0, BLOCK_D)          # d

    kc_ptr = kc + pid_b * str_kc_b + pid_h * str_kc_h + pid_i * str_kc_n_
    vc_ptr = vc + pid_b * str_vc_b + pid_h * str_vc_h + pid_i * str_vc_n_

    src_ptr = src + pid_b * str_src_b + pid_h * str_src_h + pid_i * str_src_n_ 
    src_mat = tl.load(src_ptr + offs_i * str_src_c_, mask=offs_i < BLOCK_C, other=0.0)
    src_mat = tl.cast(src_mat, tl.float32)
    src_mat = tl.exp2(tl.log2(src_mat) / 3.0)

    # Pointers that depend on j
    xq_j = xq + pid_b * str_xq_b + pid_h * str_xq_h
    kt_j = kt + pid_b * str_kt_b + pid_h * str_kt_h 
    out_j = out + pid_b * str_out_b + pid_h * str_out_h + pid_i * str_out_n_
    dest_j = dest + pid_b * str_dest_b + pid_h * str_dest_h
    denom_j = denom + pid_b * str_denom_b + pid_h * str_denom_h + pid_i * str_denom_n_ 
    offs_tri_j = pid_i * BLOCK_C

    prev_sum = tl.zeros((BLOCK_C,), dtype=tl.float32)

    for pid_j in range(0, _n):
        offs_tri = offs_tri_j - pid_j * BLOCK_R
        offs_block = pid_j * BLOCK_R + offs_j

        affinity = tl.zeros((BLOCK_C, BLOCK_R), dtype=tl.float32)

        # k_.matmul(_kt)
        kt_ptr = kt_j + pid_j * str_kt__n

        for d_offset in range(0, d, BLOCK_D):
            offs_d = d_offset + offs_k
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

        # .relu().pow(2/3)
        affinity = tl.exp2(tl.log2(tl.maximum(affinity, 0.0)) * 2.0 / 3.0)

        # * static_src_.pow(1/3).unsqueeze(-1) * static_dest.pow(1/3).unsqueeze(-2)
        dest_ptr = dest_j + pid_j * str_dest__n
        dest_mat = tl.load(dest_ptr + offs_j * str_dest__c, mask=offs_j < BLOCK_R, other=0.0)
        dest_mat = tl.cast(dest_mat, tl.float32)
        dest_mat = tl.exp2(tl.log2(dest_mat) / 3.0)

        affinity = affinity * src_mat[:, None] * dest_mat[None, :]

        # torch.log1p(affinity.clamp(min=0, max=1-1e-6).neg())
        affinity = tl.log(1.0 - tl.clamp(affinity, 0.0, 1.0 - 1e-6)) 

        # .triu(i*c_-j*_c+1)
        affinity = tl.where((offs_j[None, :] > (offs_i[:, None] + offs_tri)), affinity, 0.0)

        # .cumsum(3)
        curr_sum = tl.sum(affinity, axis=1, keep_dims=False) 
        affinity = tl.cumsum(affinity, axis=1) + prev_sum[:, None]  
        prev_sum += curr_sum

        # .masked_fill(mask.tril(i*c_-j*_c-1), -1e12)
        affinity = tl.where((offs_j[None, :] < (offs_i[:, None] + offs_tri)), -1e12, affinity)
        affinity = tl.cast(affinity, tl.float32)

        # k_.unsqueeze(2).matmul(_q.transpose(-1,-2)).add(affinity.unsqueeze(2))
        xq_ptr = xq_j + pid_j * str_xq__n
        denom_ptr = denom_j + offs_block * str_denom_l
        out_ptr = out_j + offs_block * str_out_l

        # @Haochen: storing a 3D tensor after applying .dot() to 3D tensors would fail LLVM compilation
        # The problem is fixed in triton 3.2.0, and the alternative code is listed in matmul.py
        for rep in range(0, r):
            xq_rep_ptr = xq_ptr + rep * str_xq_r
            denom_rep_ptr = denom_ptr + rep * str_denom_r
            out_rep_ptr = out_ptr + rep * str_out_r

            kq = tl.zeros((BLOCK_C, BLOCK_R), dtype=tl.float32)
            for d_offset in range(0, d, BLOCK_D):
                offs_d = d_offset + offs_k

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

            tl.store(
                denom_rep_ptr, 
                score_logsumexp, 
                mask=offs_block < BLOCK_R * _n
            )

            # score.transpose(-1,-2).softmax(dim=-1).to(dtype=_q.dtype).matmul(v_.unsqueeze(2))
            score_softmax = tl.div_rn(tl.trans(score_exp), score_sumexp[:, None])
            score_softmax = tl.cast(score_softmax, DTYPE) 
            
            for d_offset in range(0, d, BLOCK_D):
                offs_d = d_offset + offs_k

                vc_mat = tl.load(
                    vc_ptr + offs_i[:, None] * str_vc_c_ + offs_d[None, :] * str_vc_d, 
                    mask=(offs_i[:, None] < BLOCK_C) & (offs_d[None, :] < d), 
                    other=0.0
                )

                tl.store(
                    out_rep_ptr[:, None] + offs_d[None, :] * str_out_d, 
                    tl.dot(score_softmax, vc_mat, input_precision="ieee"), 
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
    dtype = xq.dtype
    device = xq.device
    DTYPE_FLAG = tl.bfloat16 if dtype == torch.bfloat16 else tl.float32

    out = torch.empty(b,h,r,l,d,n_, dtype=dtype, device=device)
    denom = torch.empty(b,h,r,l,n_, dtype=dtype, device=device)

    kt = kc.view(b,h,_n,_c,d).permute(0,1,4,2,3)  # b h d _n _c

    # Due to the sum buffer for column cumulative sum, we want to process that dimension sequencially
    grid = (b,h,n_)

    _universal_attention_fwd_kernel[grid](
        kc, vc, xq, kt, static_src, static_dest, out, denom,                                                  
        b, h, r, n_, _n, d, 
        kc.stride(0), kc.stride(1), kc.stride(2), kc.stride(3), kc.stride(4),   
        vc.stride(0), vc.stride(1), vc.stride(2), vc.stride(3), vc.stride(4),   
        xq.stride(0), xq.stride(1), xq.stride(2), xq.stride(3), xq.stride(4), xq.stride(5),  
        kt.stride(0), kt.stride(1), kt.stride(2), kt.stride(3), kt.stride(4),   
        static_src.stride(0), static_src.stride(1), static_src.stride(2), static_src.stride(3), 
        static_dest.stride(0), static_dest.stride(1), static_dest.stride(2), static_dest.stride(3), 
        out.stride(0), out.stride(1), out.stride(2), out.stride(3), out.stride(4), out.stride(5), 
        denom.stride(0), denom.stride(1), denom.stride(2), denom.stride(3), denom.stride(4), 
        BLOCK_R=_c, BLOCK_C=c_, DTYPE=DTYPE_FLAG, 
    )

    return out, denom


'''
#######################################
#     Backward Kernel & Interface     #
#######################################
'''
@triton.autotune(
    configs=configs,
    key=['r', 'n_', '_n', 'd'],
)
@triton.jit
def _universal_attention_bwd_kernel(
    # Pointers to matrices
    kc, vc, xq, kt, src, dest, dout, ddenom,    
    dkc, dvc, dxq, dsrc, ddest,
    # Matrix dimensions
    b, h, r, n_, _n, d, 
    # Strides
    str_kc_b, str_kc_h, str_kc_n_, str_kc_c_, str_kc_d,                     # b h n_ c_ d
    str_vc_b, str_vc_h, str_vc_n_, str_vc_c_, str_vc_d,                     # b h n_ c_ d
    str_xq_b, str_xq_h, str_xq_r, str_xq__n, str_xq__c, str_xq_d,           # b h r _n _c d
    str_kt_b, str_kt_h, str_kt_d, str_kt__n, str_kt__c,                     # b h d _n _c
    str_src_b, str_src_h, str_src_n_, str_src_c_,                           # b h n_ c_
    str_dest_b, str_dest_h, str_dest__n, str_dest__c,                       # b h _n _c
    str_dout_b, str_dout_h, str_dout_r, str_dout_l, str_dout_d, str_dout_n_,# b h r l d n_
    str_ddenom_b, str_ddenom_h, str_ddenom_r, str_ddenom_l, str_ddenom_n_,  # b h r l n_
    str_dkc_b, str_dkc_h, str_dkc_l, str_dkc_d,                             # b h l d
    str_dvc_b, str_dvc_h, str_dvc_n_, str_dvc_c_, str_dvc_d,                # b h n_ c_ d
    str_dxq_b, str_dxq_h, str_dxq_r, str_dxq__n, str_dxq__c, str_dxq_d,     # b h r _n _c d
    str_dsrc_b, str_dsrc_h, str_dsrc_n_, str_dsrc_c_,                       # b h n_ c_
    str_ddest_b, str_ddest_h, str_ddest__n, str_ddest__c,                   # b h _n _c
    # Meta-parameters
    BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr, BLOCK_D: tl.constexpr,    # Block dims
    DTYPE: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_i = tl.arange(0, BLOCK_C)          # c_
    offs_j = tl.arange(0, BLOCK_R)          # _c
    offs_k = tl.arange(0, BLOCK_D)          # d

    # xx_i: Pointers that depend on i
    # xx_j: Pointers that depend on j
    kc_i = kc + pid_b * str_kc_b + pid_h * str_kc_h
    kt_j = kt + pid_b * str_kt_b + pid_h * str_kt_h 
    vc_i = vc + pid_b * str_vc_b + pid_h * str_vc_h
    xq_j = xq + pid_b * str_xq_b + pid_h * str_xq_h
    src_i = src + pid_b * str_src_b + pid_h * str_src_h
    dest_j = dest + pid_b * str_dest_b + pid_h * str_dest_h
    dout_ij = dout + pid_b * str_dout_b + pid_h * str_dout_h
    ddenom_ij = ddenom + pid_b * str_ddenom_b + pid_h * str_ddenom_h

    dkc_i = dkc + pid_b * str_dkc_b + pid_h * str_dkc_h 
    dvc_i = dvc + pid_b * str_dvc_b + pid_h * str_dvc_h
    dxq_j = dxq + pid_b * str_dxq_b + pid_h * str_dxq_h
    dsrc_i = dsrc + pid_b * str_dsrc_b + pid_h * str_dsrc_h
    ddest_j = ddest + pid_b * str_ddest_b + pid_h * str_ddest_h

    # Clear out output tensors first
    for pid_i in range(0, n_):
        dkc_ptr = dkc_i + pid_i * BLOCK_C * str_dkc_l
        dvc_ptr = dvc_i + pid_i * str_dvc_n_
        dsrc_ptr = dsrc_i + pid_i * str_dsrc_n_ 
        for d_offset in range(0, d, BLOCK_D):
            offs_d = d_offset + offs_k
            tl.store(
                dkc_ptr + offs_i[:, None] * str_dkc_l + offs_d[None, :] * str_dkc_d, 
                tl.zeros((BLOCK_C, BLOCK_D), dtype=tl.float32), 
                mask=(offs_i[:, None] < BLOCK_C * n_) & (offs_d[None, :] < d), 
            ) 
            tl.store(
                dvc_ptr + offs_i[:, None] * str_dvc_c_ + offs_d[None, :] * str_dvc_d, 
                tl.zeros((BLOCK_C, BLOCK_D), dtype=tl.float32), 
                mask=(offs_i[:, None] < BLOCK_C) & (offs_d[None, :] < d), 
            )
        tl.store(
            dsrc_ptr + offs_i * str_dsrc_c_, 
            tl.zeros((BLOCK_C,), dtype=tl.float32), 
            mask=offs_i < BLOCK_C,
        )

    for pid_j in range(0, _n):
        dxq_ptr = dxq_j + pid_j * str_dxq__n
        ddest_ptr = ddest_j + pid_j * str_ddest__n
        for rep in range(0, r):
            dxq_rep_ptr = dxq_ptr + rep * str_dxq_r
            for d_offset in range(0, d, BLOCK_D):
                offs_d = d_offset + offs_k
                tl.store(
                    dxq_rep_ptr + offs_j[:, None] * str_dxq__c + offs_d[None, :] * str_dxq_d, 
                    tl.zeros((BLOCK_R, BLOCK_D), dtype=tl.float32),
                    mask=(offs_j[:, None] < BLOCK_R) & (offs_d[None, :] < d), 
                )
        tl.store(
            ddest_ptr + offs_j * str_ddest__c, 
            tl.zeros((BLOCK_R,), dtype=tl.float32), 
            mask=offs_j < BLOCK_R,
        )

    for pid_i in range(0, n_):
        kc_ptr = kc_i + pid_i * str_kc_n_ 
        vc_ptr = vc_i + pid_i * str_vc_n_ 

        src_ptr = src_i + pid_i * str_src_n_ 
        src_mat = tl.load(src_ptr + offs_i * str_src_c_, mask=offs_i < BLOCK_C, other=0.0)
        src_mat = tl.cast(src_mat, tl.float32)
        src_mat = tl.exp2(tl.log2(src_mat) / 3.0)

        dkc_ptr = dkc_i + pid_i * BLOCK_C * str_dkc_l 
        dvc_ptr = dvc_i + pid_i * str_dvc_n_
        dsrc_ptr = dsrc_i + pid_i * str_dsrc_n_ 

        dout_j = dout_ij + pid_i * str_dout_n_
        ddenom_j = ddenom_ij + pid_i * str_ddenom_n_ 
        
        offs_tri_j = pid_i * BLOCK_C

        # Clear out sum buffers
        prev_sum = tl.zeros((BLOCK_C,), dtype=tl.float32)
        daff_sum = tl.zeros((BLOCK_C,), dtype=tl.float32)

        # First forward pass
        for pid_j in range(0, _n):
            offs_tri = offs_tri_j - pid_j * BLOCK_R
            offs_block = pid_j * BLOCK_R + offs_j

            affinity = tl.zeros((BLOCK_C, BLOCK_R), dtype=tl.float32)

            # k_.matmul(_kt)
            kt_ptr = kt_j + pid_j * str_kt__n

            for d_offset in range(0, d, BLOCK_D):
                offs_d = d_offset + offs_k

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

                affinity += tl.dot(kc_mat, kt_mat, input_precision="ieee")

            # .relu().pow(2/3)
            affinity = tl.exp2(tl.log2(tl.maximum(affinity, 0.0)) * 2.0 / 3.0)

            # * static_src_.pow(1/3).unsqueeze(-1) * static_dest.pow(1/3).unsqueeze(-2)
            dest_ptr = dest_j + pid_j * str_dest__n
            dest_mat = tl.load(dest_ptr + offs_j * str_dest__c, mask=offs_j < BLOCK_R, other=0.0)
            dest_mat = tl.cast(dest_mat, tl.float32)
            dest_mat = tl.exp2(tl.log2(dest_mat) / 3.0)

            affinity = affinity * src_mat[:, None] * dest_mat[None, :]

            # torch.log1p(affinity.clamp(min=0, max=1-1e-6).neg())
            affinity = tl.log(1.0 - tl.clamp(affinity, 0.0, 1.0 - 1e-6)) 

            # .triu(i*c_-j*_c+1)
            affinity = tl.where((offs_j[None, :] > (offs_i[:, None] + offs_tri)), affinity, 0.0)

            # .cumsum(3)
            curr_sum = tl.sum(affinity, axis=1, keep_dims=False) 
            affinity = tl.cumsum(affinity, axis=1) + prev_sum[:, None]  
            prev_sum += curr_sum

            # .masked_fill(mask.tril(i*c_-j*_c-1), -1e12)
            affinity = tl.where((offs_j[None, :] < (offs_i[:, None] + offs_tri)), -1e12, affinity)
            affinity = tl.cast(affinity, tl.float32)

            # k_.unsqueeze(2).matmul(_q.transpose(-1,-2)).add(affinity.unsqueeze(2))
            xq_ptr = xq_j + pid_j * str_xq__n
            ddenom_ptr = ddenom_j + offs_block * str_ddenom_l
            dout_ptr = dout_j + offs_block * str_dout_l
            dxq_ptr = dxq_j + pid_j * str_dxq__n

            for rep in range(0, r):
                xq_rep_ptr = xq_ptr + rep * str_xq_r
                ddenom_rep_ptr = ddenom_ptr + rep * str_ddenom_r 
                dout_rep_ptr = dout_ptr + rep * str_dout_r
                dxq_rep_ptr = dxq_ptr + rep * str_dxq_r

                kq = tl.zeros((BLOCK_C, BLOCK_R), dtype=tl.float32)
                for d_offset in range(0, d, BLOCK_D):
                    offs_d = d_offset + offs_k

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

                    kq += tl.dot(kc_mat,tl.trans(xq_mat), input_precision="ieee")

                score = kq + affinity

                # Stabilize logsumexp using the subtract max trick
                score_max = tl.max(score, axis=0) 
                score_shifted = score - score_max[None, :]
                score_exp = tl.exp(score_shifted)
                score_sumexp = tl.sum(score_exp, axis=0)
                score_logsumexp = score_max + tl.log(score_sumexp)

                # score.transpose(-1,-2).softmax(dim=-1).to(dtype=_q.dtype)
                score_softmax = tl.div_rn(score_exp, score_sumexp[None, :])

                # _dscore = v_.unsqueeze(2).matmul(_dout_.transpose(-1,-2))  # b h c_ d, b h r _c d -> b h r c_ _c  (from out)
                dscore_acc = tl.zeros((BLOCK_C, BLOCK_R), dtype=tl.float32)
                for d_offset in range(0, d, BLOCK_D):
                    offs_d = d_offset + offs_k

                    vc_mat = tl.load(
                        vc_ptr + offs_i[:, None] * str_vc_c_ + offs_d[None, :] * str_vc_d, 
                        mask=(offs_i[:, None] < BLOCK_C) & (offs_d[None, :] < d), 
                        other=0.0
                    )
                    vc_mat = tl.cast(vc_mat, tl.float32)

                    dout_mat = tl.load(
                        dout_rep_ptr[:, None] + offs_d[None, :] * str_dout_d, 
                        mask=(offs_block[:, None] < BLOCK_R * _n) & (offs_d[None, :] < d), 
                        other=0.0
                    )
                    dout_mat = tl.cast(dout_mat, tl.float32)

                    dscore_acc += tl.dot(vc_mat, tl.trans(dout_mat), input_precision="ieee")
                
                # _dscore = _dscore.sub(_dscore.mul(_sscore).sum(-2,True)).mul(_sscore)  # (from softmax)
                _dscore = (dscore_acc - tl.sum(dscore_acc * score_softmax, axis=0, keep_dims=True)) * score_softmax
                # _dscore += _sscore * _ddenom_.unsqueeze(-2)  # (from denom)
                ddenom_mat = tl.load(ddenom_rep_ptr, mask=offs_block < BLOCK_R * _n, other=0.0)
                _dscore += score_softmax * ddenom_mat[None, :]
                
                # Compute the sum for the backward cumsum
                # _daff = _dscore.sum(2)  # b h c_ _c, handled via accumulating over r
                # daff_sum += _daff.sum(3)  # b h c_ 
                daff_sum += tl.sum(_dscore, axis=1)

                for d_offset in range(0, d, BLOCK_D):
                    offs_d = d_offset + offs_k

                    # dvc[:,:,i] += _sscore.to(dtype=dvc.dtype).matmul(_dout_).sum(2)  # b h r c_ _c, b h r _c d -> b h c_ d
                    dvc_mat = tl.load(
                        dvc_ptr + offs_i[:, None] * str_dvc_c_ + offs_d[None, :] * str_dvc_d, 
                        mask=(offs_i[:, None] < BLOCK_C) & (offs_d[None, :] < d), 
                        other=0.0
                    )

                    dout_mat = tl.load(
                        dout_rep_ptr[:, None] + offs_d[None, :] * str_dout_d, 
                        mask=(offs_block[:, None] < BLOCK_R * _n) & (offs_d[None, :] < d), 
                        other=0.0
                    )

                    tl.store(
                        dvc_ptr + offs_i[:, None] * str_dvc_c_ + offs_d[None, :] * str_dvc_d, 
                        dvc_mat + tl.dot(tl.cast(score_softmax, DTYPE), dout_mat, input_precision="ieee"), 
                        mask=(offs_i[:, None] < BLOCK_C) & (offs_d[None, :] < d), 
                    )

                    # dxq[:,:,:,j] += _dscore.to(dtype=dxq.dtype).transpose(-1,-2).matmul(k_.unsqueeze(2))  # b h r c_ _c, b h c_ d -> b h r _c d
                    kc_mat = tl.load(
                        kc_ptr + offs_i[:, None] * str_kc_c_ + offs_d[None, :] * str_kc_d, 
                        mask=(offs_i[:, None] < BLOCK_C) & (offs_d[None, :] < d), 
                        other=0.0
                    )

                    dxq_mat = tl.load(
                        dxq_rep_ptr + offs_j[:, None] * str_dxq__c + offs_d[None, :] * str_dxq_d, 
                        mask=(offs_j[:, None] < BLOCK_R) & (offs_d[None, :] < d), 
                        other=0.0
                    )

                    tl.store(
                        dxq_rep_ptr + offs_j[:, None] * str_dxq__c + offs_d[None, :] * str_dxq_d, 
                        dxq_mat + tl.dot(tl.trans(tl.cast(_dscore, DTYPE)), kc_mat, input_precision="ieee"),
                        mask=(offs_j[:, None] < BLOCK_R) & (offs_d[None, :] < d), 
                    )

                    # dkc[:,:,i] += _dscore.to(dtype=dkc.dtype).transpose(2,3).flatten(3,4).matmul(_q.flatten(2,3))  # b h r c_ _c, b h r _c d -> b h c_ d
                    dkc_mat = tl.load(
                        dkc_ptr + offs_i[:, None] * str_dkc_l + offs_d[None, :] * str_dkc_d, 
                        mask=(offs_i[:, None] < BLOCK_C) & (offs_d[None, :] < d), 
                        other=0.0
                    )

                    xq_mat = tl.load(
                        xq_rep_ptr + offs_j[:, None] * str_xq__c + offs_d[None, :] * str_xq_d, 
                        mask=(offs_j[:, None] < BLOCK_R) & (offs_d[None, :] < d), 
                        other=0.0
                    )

                    tl.store(
                        dkc_ptr + offs_i[:, None] * str_dkc_l + offs_d[None, :] * str_dkc_d, 
                        dkc_mat + tl.dot(tl.cast(_dscore, DTYPE), xq_mat, input_precision="ieee"),
                        mask=(offs_i[:, None] < BLOCK_C) & (offs_d[None, :] < d), 
                    )
        # First forward pass done

        # Second forward pass with gradient computes
        prev_sum = tl.zeros((BLOCK_C,), dtype=tl.float32)
        prev_dsum = tl.zeros((BLOCK_C,), dtype=tl.float32) # sum buffer for backward 
        for pid_j in range(0, _n):
            offs_tri = offs_tri_j - pid_j * BLOCK_R
            offs_block = pid_j * BLOCK_R + offs_j
            ddest_ptr = ddest_j + pid_j * str_ddest__n

            # k_.matmul(_kt)
            kt_ptr = kt_j + pid_j * str_kt__n

            affinity = tl.zeros((BLOCK_C, BLOCK_R), dtype=tl.float32)

            for d_offset in range(0, d, BLOCK_D):
                offs_d = d_offset + offs_k

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

                affinity += tl.dot(kc_mat, kt_mat, input_precision="ieee")

            _aff1 = affinity

            # .relu().pow(2/3)
            affinity = tl.exp2(tl.log2(tl.maximum(affinity, 0.0)) * 2.0 / 3.0)

            # * static_src_.pow(1/3).unsqueeze(-1) * static_dest.pow(1/3).unsqueeze(-2)
            dest_ptr = dest_j + pid_j * str_dest__n
            dest_mat = tl.load(dest_ptr + offs_j * str_dest__c, mask=offs_j < BLOCK_R, other=0.0)
            dest_mat = tl.cast(dest_mat, tl.float32)
            dest_mat = tl.exp2(tl.log2(dest_mat) / 3.0)

            affinity = affinity * src_mat[:, None] * dest_mat[None, :]

            _aff2 = affinity

            # torch.log1p(affinity.clamp(min=0, max=1-1e-6).neg())
            affinity = tl.log(1.0 - tl.clamp(affinity, 0.0, 1.0 - 1e-6)) 

            # .triu(i*c_-j*_c+1)
            affinity = tl.where((offs_j[None, :] > (offs_i[:, None] + offs_tri)), affinity, 0.0)

            # .cumsum(3)
            curr_sum = tl.sum(affinity, axis=1, keep_dims=False) 
            affinity = tl.cumsum(affinity, axis=1) + prev_sum[:, None]  
            prev_sum += curr_sum

            # .masked_fill(mask.tril(i*c_-j*_c-1), -1e12)
            affinity = tl.where((offs_j[None, :] < (offs_i[:, None] + offs_tri)), -1e12, affinity)
            affinity = tl.cast(affinity, tl.float32)

            # k_.unsqueeze(2).matmul(_q.transpose(-1,-2)).add(affinity.unsqueeze(2))
            xq_ptr = xq_j + pid_j * str_xq__n
            ddenom_ptr = ddenom_j + offs_block * str_ddenom_l
            dout_ptr = dout_j + offs_block * str_dout_l
            dxq_ptr = dxq_j + pid_j * str_dxq__n

            _daff = tl.zeros((BLOCK_C, BLOCK_R), dtype=tl.float32)

            for rep in range(0, r):
                xq_rep_ptr = xq_ptr + rep * str_xq_r
                ddenom_rep_ptr = ddenom_ptr + rep * str_ddenom_r 
                dout_rep_ptr = dout_ptr + rep * str_dout_r
                dxq_rep_ptr = dxq_ptr + rep * str_dxq_r

                kq = tl.zeros((BLOCK_C, BLOCK_R), dtype=tl.float32)

                for d_offset in range(0, d, BLOCK_D):
                    offs_d = d_offset + offs_k

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

                    kq += tl.dot(kc_mat,tl.trans(xq_mat), input_precision="ieee")

                score = kq + affinity

                # Stabilize logsumexp using the subtract max trick
                score_max = tl.max(score, axis=0) 
                score_shifted = score - score_max[None, :]
                score_exp = tl.exp(score_shifted)
                score_sumexp = tl.sum(score_exp, axis=0)
                score_logsumexp = score_max + tl.log(score_sumexp)

                # score.transpose(-1,-2).softmax(dim=-1).to(dtype=_q.dtype).matmul(v_.unsqueeze(2))
                score_softmax = tl.div_rn(score_exp, score_sumexp[None, :])
                score_softmax = tl.cast(score_softmax, DTYPE) 

                # _dscore = v_.unsqueeze(2).matmul(_dout_.transpose(-1,-2))  # b h c_ d, b h r _c d -> b h r c_ _c  (from out)
                dscore_acc = tl.zeros((BLOCK_C, BLOCK_R), dtype=tl.float32)

                for d_offset in range(0, d, BLOCK_D):
                    offs_d = d_offset + offs_k

                    vc_mat = tl.load(
                        vc_ptr + offs_i[:, None] * str_vc_c_ + offs_d[None, :] * str_vc_d, 
                        mask=(offs_i[:, None] < BLOCK_C) & (offs_d[None, :] < d), 
                        other=0.0
                    )
                    vc_mat = tl.cast(vc_mat, tl.float32)

                    dout_mat = tl.load(
                        dout_rep_ptr[:, None] + offs_d[None, :] * str_dout_d, 
                        mask=(offs_block[:, None] < BLOCK_R * _n) & (offs_d[None, :] < d), 
                        other=0.0
                    )
                    dout_mat = tl.cast(dout_mat, tl.float32)

                    dscore_acc += tl.dot(vc_mat, tl.trans(dout_mat), input_precision="ieee")
                
                # _dscore = _dscore.sub(_dscore.mul(_sscore).sum(-2,True)).mul(_sscore)  # (from softmax)
                _dscore = (dscore_acc - tl.sum(dscore_acc * score_softmax, axis=0, keep_dims=True)) * score_softmax
                # _dscore += _sscore * _ddenom_.unsqueeze(-2)  # (from denom)
                ddenom_mat = tl.load(ddenom_rep_ptr, mask=offs_block < BLOCK_R * _n, other=0.0)
                _dscore += score_softmax * ddenom_mat[None, :]
                
                # Compute the sum for the backward cumsum
                # _daff = _dscore.sum(2)  # b h c_ _c
                _daff += _dscore
            
            # _daff_cs = _daff.cumsum(3)  # (from cumsum)
            # _daff_cs += dsum_buffer.unsqueeze(-1)   # Accumulate across row chunks
            # dsum_buffer = _daff_cs[:,:,:,-1].clone()  # Cloning prevents subsequent _aff2 calcs from changing buffer values                
            curr_dsum = tl.sum(_daff, axis=1, keep_dims=False) 
            _daff_cs = tl.cumsum(_daff, axis=1) + prev_dsum[:, None]  
            prev_dsum += curr_dsum
            # _daff += daff_sum.unsqueeze(-1) - _daff_cs
            _daff = daff_sum[:, None] - _daff_cs + _daff

            # _daff = _daff.triu(i*c_-j*_c+1)  # Prevent accumulation into masked regions
            _daff = tl.where((offs_j[None, :] > (offs_i[:, None] + offs_tri)), _daff, 0.0)

            # _daff /= _aff2.clamp(min=1e-6, max=1-1e-6) - 1  # ( from ln(1-x) )
            _daff = tl.div_rn(_daff, (tl.clamp(_aff2, 1e-6, 1.0 - 1e-6) - 1.0))

            # _daff *= _aff2.le(1-1e-6)
            _daff = tl.where((_aff2 <= (1.0 - 1e-6)), _daff, 0.0)

            # _dstat = _daff.mul(_aff1.relu().pow(2/3)).to(dtype=static_src.dtype)  # b h c_ _c
            _dstat = _daff * tl.exp2(tl.log2(tl.maximum(_aff1, 0.0)) * 2.0 / 3.0)
            _dstat = tl.cast(_dstat, tl.float32)

            # Backprop into stat_src and stat_dest            
            # dstat_src[:,:,i] += _dstat.mul(_static_dest.unsqueeze(-2)).sum(-1).div(static_src_.pow(2).mul(3))  # b h c_ _c, b h _c -> b h c_
            dsrc_mat = tl.load(dsrc_ptr + offs_i * str_dsrc_c_, mask=offs_i < BLOCK_C, other=0.0)
            dsrc_mat += tl.div_rn(tl.sum(_dstat * dest_mat[None, :], axis=1), tl.exp2(tl.log2(src_mat) * 2.0) * 3.0)
            tl.store(
                dsrc_ptr + offs_i * str_dsrc_c_, 
                dsrc_mat, 
                mask=offs_i < BLOCK_C
            )

            # dstat_dest[:,:,j] += _dstat.mul(static_src_.unsqueeze(-1)).sum(-2).div(_static_dest.pow(2).mul(3))  # b h c_ _c, b h c_ -> b h _c
            ddest_mat = tl.load(ddest_ptr + offs_j * str_ddest__c, mask=offs_j < BLOCK_R, other=0.0)
            ddest_mat += tl.div_rn(tl.sum(_dstat * src_mat[:, None], axis=0), tl.exp2(tl.log2(dest_mat) * 2.0) * 3.0)
            tl.store(
                ddest_ptr + offs_j * str_ddest__c, 
                ddest_mat, 
                mask=offs_j < BLOCK_R
            )

            # # Backprop into k/k matmul
            # _daff *= static_src_.unsqueeze(-1) * _static_dest.unsqueeze(-2)  # (from prod with statics)
            _daff = _daff * src_mat[:, None] * dest_mat[None, :]
            
            # _daff = _daff.to(dtype=_q.dtype) * _aff1.abs().add(1e-9).pow(-1/3).mul(2/3).mul(_aff1.gt(0))  # (from relu and pow)
            _daff *= tl.exp2(-tl.log2((tl.abs(_aff1) + 1e-9)) / 3.0) * (2.0 / 3.0)
            _daff = tl.where((_aff1 > 0.0), _daff, 0.0)
            _daff = tl.cast(_daff, DTYPE)

            for d_offset in range(0, d, BLOCK_D):
                offs_d = d_offset + offs_k

                # dkc[:,:,j*_c:(j+1)*_c] += _daff.transpose(-1,-2).matmul(k_)  # b h c_ _c, b h c_ d -> b h _c d
                dkc_j_ptr = dkc_i + pid_j * BLOCK_R * str_dkc_l 
                dkc_j_mat = tl.load(
                    dkc_j_ptr + offs_j[:, None] * str_dkc_l + offs_d[None, :] * str_dkc_d, 
                    mask=(offs_j[:, None] < BLOCK_R) & (offs_d[None, :] < d), 
                    other=0.0
                )
                # dkc_j_mat = tl.cast(dkc_j_mat, tl.float32)
                
                kc_mat = tl.load(
                    kc_ptr + offs_i[:, None] * str_kc_c_ + offs_d[None, :] * str_kc_d, 
                    mask=(offs_i[:, None] < BLOCK_C) & (offs_d[None, :] < d), 
                    other=0.0
                )
                # kc_mat = tl.cast(kc_mat, tl.float32)

                tl.store(
                    dkc_j_ptr + offs_j[:, None] * str_dkc_l + offs_d[None, :] * str_dkc_d, 
                    dkc_j_mat + tl.dot(tl.trans(_daff), kc_mat, input_precision="ieee"),
                    mask=(offs_j[:, None] < BLOCK_R) & (offs_d[None, :] < d), 
                )
                
                # dkc[:,:,i*c_:(i+1)*c_] += _daff.matmul(_kt.transpose(-1,-2))  # b h c_ _c, b h d _c -> b h c_ d
                dkc_mat = tl.load(
                    dkc_ptr + offs_i[:, None] * str_dkc_l + offs_d[None, :] * str_dkc_d, 
                    mask=(offs_i[:, None] < BLOCK_C) & (offs_d[None, :] < d), 
                    other=0.0
                )
                # dkc_mat = tl.cast(dkc_mat, tl.float32)

                kt_mat = tl.load(
                    kt_ptr + offs_d[:, None] * str_kt_d + offs_j[None, :] * str_kt__c, 
                    mask=(offs_d[:, None] < d) & (offs_j[None, :] < BLOCK_R), 
                    other=0.0
                )
                # kt_mat = tl.cast(kt_mat, tl.float32)

                tl.store(
                    dkc_ptr + offs_i[:, None] * str_dkc_l + offs_d[None, :] * str_dkc_d, 
                    dkc_mat + tl.dot(_daff, tl.trans(kt_mat), input_precision="ieee"),
                    mask=(offs_i[:, None] < BLOCK_C) & (offs_d[None, :] < d), 
                )

def _universal_attention_bwd(kc, vc, xq, static_src, static_dest, dout, ddenom):
    '''
    Inputs:
    kc: b h n_ c_ d
    vc: b h n_ c_ d
    xq: b h r _n _c d
    static_src: b h n_ c_
    static_dest: b h _n _c
    dout: b h r l d n_
    ddenom: b h r l n_

    Outputs:
    dkc: b h n_ c_ d
    dvc: b h n_ c_ d
    dxq: b h r _n _c d
    dstatic_src: b h n_ c_
    dstatic_dest: b h _n _c

    Intermediate: 
    variables_ are split over cols
    _variables are split over rows
    (i.e. n_ is the number of col chunks, _n is the number of row chunks)

    Note: when using mixed precision, dout is downcasted but ddenom is always fp32
    '''
    b,h,r,_n,_c,d = xq.shape
    _,_,n_,c_,_ = kc.shape
    l = n_*c_
    dtype = xq.dtype
    device = xq.device
    DTYPE_FLAG = tl.bfloat16 if dtype == torch.bfloat16 else tl.float32

    dkc = torch.empty(b,h,l,d, dtype=dtype, device=device)
    dvc = torch.empty(b,h,n_,c_,d, dtype=dtype, device=device)
    dxq = torch.empty(b,h,r,_n,_c,d, dtype=dtype, device=device)
    dstatic_src = torch.empty(b,h,n_,c_, dtype=static_src.dtype, device=device)
    dstatic_dest = torch.empty(b,h,_n,_c, dtype=dtype, device=device)

    kt = kc.view(b,h,_n,_c,d).permute(0,1,4,2,3)  # b h d _n _c
    # The total size of the buffers are O(n^2), so it would probably better if we can avoid using them
    # sum_buffer = torch.empty(b,h,n_,c_, dtype=static_src.dtype, device=static_src.device)
    # aff1 = torch.empty(b,h,n_,c_,l, dtype=dtype, device=device)
    # sscore = torch.empty(b,h,r,n_,c_,l, dtype=dtype, device=device)

    # Due to the sum buffer for column cumulative sum, we want to process that dimension sequencially
    # And since the backward pass needs accumulation on the rows, we want to process that dimension sequencially as well
    grid = (b,h)

    _universal_attention_bwd_kernel[grid](
        kc, vc, xq, kt, static_src, static_dest, dout, ddenom,
        dkc, dvc, dxq, dstatic_src, dstatic_dest,
        b, h, r, n_, _n, d, 
        kc.stride(0), kc.stride(1), kc.stride(2), kc.stride(3), kc.stride(4),  
        vc.stride(0), vc.stride(1), vc.stride(2), vc.stride(3), vc.stride(4),   
        xq.stride(0), xq.stride(1), xq.stride(2), xq.stride(3), xq.stride(4), xq.stride(5),  
        kt.stride(0), kt.stride(1), kt.stride(2), kt.stride(3), kt.stride(4),   
        static_src.stride(0), static_src.stride(1), static_src.stride(2), static_src.stride(3), 
        static_dest.stride(0), static_dest.stride(1), static_dest.stride(2), static_dest.stride(3), 
        dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3), dout.stride(4), dout.stride(5), 
        ddenom.stride(0), ddenom.stride(1), ddenom.stride(2), ddenom.stride(3), ddenom.stride(4), 
        dkc.stride(0), dkc.stride(1), dkc.stride(2), dkc.stride(3),
        dvc.stride(0), dvc.stride(1), dvc.stride(2), dvc.stride(3), dvc.stride(4), 
        dxq.stride(0), dxq.stride(1), dxq.stride(2), dxq.stride(3), dxq.stride(4), dxq.stride(5), 
        dstatic_src.stride(0), dstatic_src.stride(1), dstatic_src.stride(2), dstatic_src.stride(3), 
        dstatic_dest.stride(0), dstatic_dest.stride(1), dstatic_dest.stride(2), dstatic_dest.stride(3), 
        BLOCK_R=_c, BLOCK_C=c_, DTYPE=DTYPE_FLAG, 
    )
    dkc = dkc.view(b,h,n_,c_,d)

    return dkc, dvc, dxq, dstatic_src, dstatic_dest


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
            #print(f'score-torch: {torch.nan_to_num(score).sum()}')
            #print(f'qk-torch: {torch.nan_to_num(k_.unsqueeze(2).matmul(_q.transpose(-1,-2))).sum()}')
            #print(f'affinity-torch: {affinity}')
            #print(f'qk-torch: {k_.unsqueeze(2).matmul(_q.transpose(-1,-2)).sum()}')
            #print(f'score-torch: {score.sum()}')
            _denom_ = score.logsumexp(dim=-2)  # b h r _c
            _out_ = score.transpose(-1,-2).softmax(dim=-1).to(dtype=_q.dtype).matmul(v_.unsqueeze(2))  # b h r _c d
            #print(f'torch-attn: {_out_.sum()}')

            out[:,:,:,j*_c:(j+1)*_c,:,i] = _out_
            denom[:,:,:,j*_c:(j+1)*_c,i] = _denom_
    return out, denom

def universal_attention_backward_single_direction(kc, vc, xq, static_src, static_dest, dout, ddenom):
    '''
    Inputs:
    kc: b h n_ c_ d
    vc: b h n_ c_ d
    xq: b h r _n _c d
    static_src: b h n_ c_
    static_dest: b h _n _c
    dout: b h r l d n_
    ddenom: b h r l n_

    Outputs:
    dkc: b h n_ c_ d
    dvc: b h n_ c_ d
    dxq: b h r _n _c d
    dstatic_src: b h n_ c_
    dstatic_dest: b h _n _c

    Intermediate: 
    variables_ are split over cols
    _variables are split over rows
    (i.e. n_ is the number of col chunks, _n is the number of row chunks)

    Note: when using mixed precision, dout is downcasted but ddenom is always fp32
    '''
    # Single direction kernel time: 1645.0271136760712s
    # Original kernel time: 1312.5504279136658s
    dkc,dvc,dxq,dstat_src,dstat_dest = [torch.zeros_like(x) for x in [kc,vc,xq,static_src,static_dest]]
    dxq1 = torch.zeros_like(dxq)
    b,h,r,_n,_c,d = xq.shape
    _,_,n_,c_,_ = kc.shape
    l = n_ * c_
    mask = torch.ones(c_,_c, dtype=torch.bool, device=xq.device)
    static_src = static_src.pow(1/3)
    static_dest = static_dest.pow(1/3)
    kt = kc.view(b,h,_n,_c,d).permute(0,1,4,2,3)  # b h d _n _c

    # Iterate over columns
    # Removed the necessity for loading/storing intermediate tensors
    # Now every save/load between SRAM and HBM are made to load input or store final results
    sum_buffer = torch.empty(b,h,c_, dtype=static_src.dtype, device=static_src.device)
    dsum_buffer = torch.empty(b,h,c_, dtype=static_src.dtype, device=static_src.device)

    for i in range(n_):
        k_ = kc[:,:,i]  # b h c_ d
        v_ = vc[:,:,i]  # b h c_ d
        static_src_ = static_src[:,:,i]  # b h c_
        dout_ = dout[...,i]  # b h r l d
        ddenom_ = ddenom[...,i]  # b h r l

        # Rerun forward pass
        daff_sum = torch.zeros(b,h,c_, dtype=torch.float, device=k_.device)

        sum_buffer.zero_()
        # Iterate over rows
        for j in range(_n):
            _q = xq[:,:,:,j]  # b h r _c d
            _static_dest = static_dest[:,:,j]  # b h _c
            _kt = kt[:,:,:,j]  # b h d _c
            affinity = k_.matmul(_kt)  # b h c_ _c
            affinity = affinity.relu().pow(2/3).float() * static_src_.unsqueeze(-1) * _static_dest.unsqueeze(-2)
            affinity = torch.log1p(affinity.clamp(min=0, max=1-1e-6).neg())  # b h c_ _c
            affinity = affinity.triu(i*c_-j*_c+1).cumsum(3)  # Accumulate decay with causal masking, within row chunk
            affinity += sum_buffer.unsqueeze(-1)  # Accumulate decay from prior row chunk
            sum_buffer = affinity[:,:,:,-1]
            affinity = affinity.masked_fill(mask.tril(i*c_-j*_c-1), -1e12)  # Re-mask, with 1s on diagonal
            _sscore = k_.unsqueeze(2).matmul(_q.transpose(-1,-2)).add(affinity.unsqueeze(2)).softmax(dim=-2)  # b h r c_ _c

            _dout_ = dout_[:,:,:,j*_c:(j+1)*_c]  # b h r _c d
            _ddenom_ = ddenom_[:,:,:,j*_c:(j+1)*_c]  # b h r _c

            dvc[:,:,i] += _sscore.to(dtype=dvc.dtype).matmul(_dout_).sum(2)  # b h r c_ _c, b h r _c d -> b h c_ d
            _dscore = v_.unsqueeze(2).matmul(_dout_.transpose(-1,-2))  # b h c_ d, b h r _c d -> b h r c_ _c  (from out)
            _dscore = _dscore.sub(_dscore.mul(_sscore).sum(-2,True)).mul(_sscore)  # (from softmax)
            _dscore += _sscore * _ddenom_.unsqueeze(-2)  # (from denom)

            # Backprop through q/k matmul
            dxq[:,:,:,j] += _dscore.to(dtype=dxq.dtype).transpose(-1,-2).matmul(k_.unsqueeze(2))  # b h r c_ _c, b h c_ d -> b h r _c d            
            dkc[:,:,i] += _dscore.to(dtype=dkc.dtype).transpose(2,3).flatten(3,4).matmul(_q.flatten(2,3))  # b h r c_ _c, b h r _c d -> b h c_ d

            # Compute the sum for the backward cumsum
            _daff = _dscore.sum(2)  # b h c_ _c
            daff_sum += _daff.sum(3)  # b h c_ 

        # Backward pass
        sum_buffer.zero_()
        dsum_buffer.zero_()

        # Iterate over rows, backward
        for j in range(_n):
            _q = xq[:,:,:,j]  # b h r _c d
            _static_dest = static_dest[:,:,j]  # b h _c
            _kt = kt[:,:,:,j]  # b h d _c
            _dout_ = dout_[:,:,:,j*_c:(j+1)*_c]  # b h r _c d
            _ddenom_ = ddenom_[:,:,:,j*_c:(j+1)*_c]  # b h r _c

            affinity = k_.matmul(_kt)  # b h c_ _c
            _aff1 = affinity.clone()
            affinity = affinity.relu().pow(2/3).float() * static_src_.unsqueeze(-1) * _static_dest.unsqueeze(-2)
            _aff2 = affinity.clone()
            affinity = torch.log1p(affinity.clamp(min=0, max=1-1e-6).neg())  # b h c_ _c
            affinity = affinity.triu(i*c_-j*_c+1).cumsum(3)  # Accumulate decay with causal masking, within row chunk
            affinity += sum_buffer.unsqueeze(-1)  # Accumulate decay from prior row chunk
            sum_buffer = affinity[:,:,:,-1]
            affinity = affinity.masked_fill(mask.tril(i*c_-j*_c-1), -1e12)  # Re-mask, with 1s on diagonal
            
            _sscore = k_.unsqueeze(2).matmul(_q.transpose(-1,-2)).add(affinity.unsqueeze(2)).softmax(dim=-2)  # b h r c_ _c

            # Backprop through score/v matmul
            _dscore = v_.unsqueeze(2).matmul(_dout_.transpose(-1,-2))  # b h c_ d, b h r _c d -> b h r c_ _c  (from out)
            _dscore = _dscore.sub(_dscore.mul(_sscore).sum(-2,True)).mul(_sscore)  # (from softmax)
            _dscore += _sscore * _ddenom_.unsqueeze(-2)  # (from denom)

            # Backprop through affinity matrix
            _daff = _dscore.sum(2)  # b h c_ _c
            _daff_cs = _daff.cumsum(3)  # (from cumsum)
            _daff_cs += dsum_buffer.unsqueeze(-1)   # Accumulate across row chunks
            dsum_buffer = _daff_cs[:,:,:,-1].clone()  # Cloning prevents subsequent _aff2 calcs from changing buffer values                
            _daff += daff_sum.unsqueeze(-1) - _daff_cs
            _daff = _daff.triu(i*c_-j*_c+1)  # Prevent accumulation into masked regions

            _daff /= _aff2.clamp(min=1e-6, max=1-1e-6) - 1  # ( from ln(1-x) )
            _daff *= _aff2.le(1-1e-6)
            _dstat = _daff.mul(_aff1.relu().pow(2/3)).to(dtype=static_src.dtype)  # b h c_ _c

            # Backprop into stat_src and stat_dest
            dstat_src[:,:,i] += _dstat.mul(_static_dest.unsqueeze(-2)).sum(-1).div(static_src_.pow(2).mul(3))  # b h c_ _c, b h _c -> b h c_
            dstat_dest[:,:,j] += _dstat.mul(static_src_.unsqueeze(-1)).sum(-2).div(_static_dest.pow(2).mul(3))  # b h c_ _c, b h c_ -> b h _c

            # Backprop into k/k matmul
            _daff *= static_src_.unsqueeze(-1) * _static_dest.unsqueeze(-2)  # (from prod with statics)
            _daff = _daff.to(dtype=_q.dtype) * _aff1.abs().add(1e-9).pow(-1/3).mul(2/3).mul(_aff1.gt(0))  # (from relu and pow)
            dkc = dkc.view(b,h,l,d)
            dkc[:,:,j*_c:(j+1)*_c] += _daff.transpose(-1,-2).matmul(k_)  # b h c_ _c, b h c_ d -> b h _c d
            dkc[:,:,i*c_:(i+1)*c_] += _daff.matmul(_kt.transpose(-1,-2))  # b h c_ _c, b h d _c -> b h c_ d
            dkc = dkc.view(b,h,n_,c_,d)

    return dkc,dvc,dxq,dstat_src,dstat_dest


def universal_attention_backward(kc, vc, xq, static_src, static_dest, dout, ddenom):
    '''
    Inputs:
    kc: b h n_ c_ d
    vc: b h n_ c_ d
    xq: b h r _n _c d
    static_src: b h n_ c_
    static_dest: b h _n _c
    dout: b h r l d n_
    ddenom: b h r l n_

    Outputs:
    dkc: b h n_ c_ d
    dvc: b h n_ c_ d
    dxq: b h r _n _c d
    dstatic_src: b h n_ c_
    dstatic_dest: b h _n _c

    Intermediate: 
    variables_ are split over cols
    _variables are split over rows
    (i.e. n_ is the number of col chunks, _n is the number of row chunks)

    Note: when using mixed precision, dout is downcasted but ddenom is always fp32
    '''
    dkc,dvc,dxq,dstat_src,dstat_dest = [torch.zeros_like(x) for x in [kc,vc,xq,static_src,static_dest]]
    b,h,r,_n,_c,d = xq.shape
    _,_,n_,c_,_ = kc.shape
    l = n_ * c_
    mask = torch.ones(c_,_c, dtype=torch.bool, device=xq.device)
    static_src = static_src.pow(1/3)
    static_dest = static_dest.pow(1/3)
    kt = kc.view(b,h,_n,_c,d).permute(0,1,4,2,3)  # b h d _n _c

    # Iterate over columns
    sum_buffer = torch.empty(b,h,c_, dtype=static_src.dtype, device=static_src.device)
    for i in range(n_):
        k_ = kc[:,:,i]  # b h c_ d
        v_ = vc[:,:,i]  # b h c_ d
        static_src_ = static_src[:,:,i]  # b h c_
        dout_ = dout[...,i]  # b h r l d
        ddenom_ = ddenom[...,i]  # b h r l

        # Rerun forward pass
        aff1 = torch.empty(b,h,c_,l, dtype=k_.dtype, device=k_.device)
        sscore = torch.empty(b,h,r,c_,l, dtype=torch.float, device=k_.device)
        sum_buffer.zero_()
        # Iterate over rows
        for j in range(_n):
            _q = xq[:,:,:,j]  # b h r _c d
            _static_dest = static_dest[:,:,j]  # b h _c
            _kt = kt[:,:,:,j]  # b h d _c
            affinity = k_.matmul(_kt)  # b h c_ _c
            aff1[:,:,:,j*_c:(j+1)*_c] = affinity
            affinity = affinity.relu().pow(2/3).float() * static_src_.unsqueeze(-1) * _static_dest.unsqueeze(-2)
            affinity = torch.log1p(affinity.clamp(min=0, max=1-1e-6).neg())  # b h c_ _c
            affinity = affinity.triu(i*c_-j*_c+1).cumsum(3)  # Accumulate decay with causal masking, within row chunk
            affinity += sum_buffer.unsqueeze(-1)  # Accumulate decay from prior row chunk
            sum_buffer = affinity[:,:,:,-1]
            affinity = affinity.masked_fill(mask.tril(i*c_-j*_c-1), -1e12)  # Re-mask, with 1s on diagonal
            sscore[:,:,:,:,j*_c:(j+1)*_c] = k_.unsqueeze(2).matmul(_q.transpose(-1,-2)).add(affinity.unsqueeze(2)).softmax(dim=-2)  # b h r c_ _c

            
        # Backward pass
        sum_buffer.zero_()
        # Iterate over rows, backward
        for j in range(_n-1,-1,-1):
            _q = xq[:,:,:,j]  # b h r _c d
            _static_dest = static_dest[:,:,j]  # b h _c
            _kt = kt[:,:,:,j]  # b h d _c
            _aff1 = aff1[:,:,:,j*_c:(j+1)*_c]  # b h c_ _c
            _aff2 = _aff1.relu().pow(2/3).float() * static_src_.unsqueeze(-1) * _static_dest.unsqueeze(-2)
            _sscore = sscore[:,:,:,:,j*_c:(j+1)*_c]  # b h r c_ _c
            _dout_ = dout_[:,:,:,j*_c:(j+1)*_c]  # b h r _c d
            _ddenom_ = ddenom_[:,:,:,j*_c:(j+1)*_c]  # b h r _c

            # Backprop through score/v matmul
            dvc[:,:,i] += _sscore.to(dtype=dvc.dtype).matmul(_dout_).sum(2)  # b h r c_ _c, b h r _c d -> b h c_ d
            _dscore = v_.unsqueeze(2).matmul(_dout_.transpose(-1,-2))  # b h c_ d, b h r _c d -> b h r c_ _c  (from out)
            _dscore = _dscore.sub(_dscore.mul(_sscore).sum(-2,True)).mul(_sscore)  # (from softmax)
            _dscore += _sscore * _ddenom_.unsqueeze(-2)  # (from denom)

            # Backprop through q/k matmul
            dxq[:,:,:,j] += _dscore.to(dtype=dxq.dtype).transpose(-1,-2).matmul(k_.unsqueeze(2))  # b h r c_ _c, b h c_ d -> b h r _c d
            dkc[:,:,i] += _dscore.to(dtype=dkc.dtype).transpose(2,3).flatten(3,4).matmul(_q.flatten(2,3))  # b h r c_ _c, b h r _c d -> b h c_ d

            # Backprop through affinity matrix
            _daff = _dscore.sum(2)  # b h c_ _c
            _daff = _daff.flip([3]).cumsum(3).flip([3])  # (from cumsum)
            _daff += sum_buffer.unsqueeze(-1)  # Accumulate across row chunks
            _daff = _daff.triu(i*c_-j*_c+1)  # Prevent accumulation into masked regions
            sum_buffer = _daff[:,:,:,0].clone()  # Cloning prevents subsequent _aff2 calcs from changing buffer values                
            
            _daff /= _aff2.clamp(min=1e-6, max=1-1e-6) - 1  # ( from ln(1-x) )
            _daff *= _aff2.le(1-1e-6)
            _dstat = _daff.mul(_aff1.relu().pow(2/3)).to(dtype=static_src.dtype)  # b h c_ _c

            # Backprop into stat_src and stat_dest
            dstat_src[:,:,i] += _dstat.mul(_static_dest.unsqueeze(-2)).sum(-1).div(static_src_.pow(2).mul(3))  # b h c_ _c, b h _c -> b h c_
            dstat_dest[:,:,j] += _dstat.mul(static_src_.unsqueeze(-1)).sum(-2).div(_static_dest.pow(2).mul(3))  # b h c_ _c, b h c_ -> b h _c

            # Backprop into k/k matmul
            _daff *= static_src_.unsqueeze(-1) * _static_dest.unsqueeze(-2)  # (from prod with statics)
            _daff = _daff.to(dtype=_q.dtype) * _aff1.abs().add(1e-9).pow(-1/3).mul(2/3).mul(_aff1.gt(0))  # (from relu and pow)
            dkc = dkc.view(b,h,l,d)
            dkc[:,:,j*_c:(j+1)*_c] += _daff.transpose(-1,-2).matmul(k_)  # b h c_ _c, b h c_ d -> b h _c d
            dkc[:,:,i*c_:(i+1)*c_] += _daff.matmul(_kt.transpose(-1,-2))  # b h c_ _c, b h d _c -> b h c_ d
            dkc = dkc.view(b,h,n_,c_,d)

    return dkc,dvc,dxq,dstat_src,dstat_dest


if __name__ == "__main__":
    # b, h, r, n_, c_, _n, _c, d = 1, 1, 1, 4, 16, 4, 16, 32
    b, h, r, n_, c_, _n, _c, d = 2, 4, 2, 32, 64, 32, 64, 512
    dtype = torch.bfloat16

    # test = "forward"
    test = "backward"
    # test = "quick"

    if test == "forward":
        print("Testing forward pass")
        kc = torch.rand((b, h, n_, c_, d), device='cuda', dtype=dtype)
        vc = torch.rand((b, h, n_, c_, d), device='cuda', dtype=dtype)
        xq = torch.rand((b, h, r, _n, _c, d), device='cuda', dtype=dtype)
        static_src = torch.rand((b, h, n_, c_), device='cuda', dtype=dtype)
        static_dest = torch.rand((b, h, _n, _c), device='cuda', dtype=dtype)

        warm_up = 10
        for _ in range(warm_up):
            _, _ = _universal_attention_fwd(kc, vc, xq, static_src, static_dest)
            _, _ = universal_attention_forward(kc, vc, xq, static_src, static_dest)

        print("Checking running time...")
        n = 1000
        triton_time, torch_time = 0, 0
        for _ in range(n):
            kc = torch.rand((b, h, n_, c_, d), device='cuda', dtype=dtype)
            vc = torch.rand((b, h, n_, c_, d), device='cuda', dtype=dtype)
            xq = torch.rand((b, h, r, _n, _c, d), device='cuda', dtype=dtype)
            static_src = torch.rand((b, h, n_, c_), device='cuda', dtype=dtype)
            static_dest = torch.rand((b, h, _n, _c), device='cuda', dtype=dtype)

            start_time = time.time()
            _, _ = _universal_attention_fwd(kc, vc, xq, static_src, static_dest)
            triton_time += time.time() - start_time

            start_time = time.time()
            _, _ = universal_attention_forward(kc, vc, xq, static_src, static_dest)
            torch_time += time.time() - start_time

            del kc, vc, xq, static_src, static_dest

        print(f"Triton kernel time: {triton_time}s")
        print(f"Pytorch kernel time: {torch_time}s")

        print("Checking closeness to ground truth...")

        kc = torch.rand((b, h, n_, c_, d), device='cuda', dtype=dtype)
        vc = torch.rand((b, h, n_, c_, d), device='cuda', dtype=dtype)
        xq = torch.rand((b, h, r, _n, _c, d), device='cuda', dtype=dtype)
        static_src = torch.rand((b, h, n_, c_), device='cuda', dtype=dtype)
        static_dest = torch.rand((b, h, _n, _c), device='cuda', dtype=dtype)

        out, denom = _universal_attention_fwd(kc, vc, xq, static_src, static_dest)
        out_ref, denom_ref = universal_attention_forward(kc, vc, xq, static_src, static_dest)

        # Convert to final output for sanity check
        output = out.mul(denom.softmax(dim=-1).unsqueeze(-2)).sum(-1)
        output_ref = out_ref.mul(denom_ref.softmax(dim=-1).unsqueeze(-2)).sum(-1)

        print("Checking denom:")
        torch.testing.assert_close(denom, denom_ref, atol=1e-3, rtol=1e-5)
        print("Checking out:")
        torch.testing.assert_close(out, out_ref, atol=1e-3, rtol=1e-5)
        print("Checking output:")
        torch.testing.assert_close(output, output_ref, atol=1e-3, rtol=1e-5)
    
    elif test == "backward":
        print("Testing backward pass")
        kc = torch.rand((b, h, n_, c_, d), device='cuda', dtype=dtype)
        vc = torch.rand((b, h, n_, c_, d), device='cuda', dtype=dtype)
        xq = torch.rand((b, h, r, _n, _c, d), device='cuda', dtype=dtype)
        static_src = torch.rand((b, h, n_, c_), device='cuda', dtype=dtype)
        static_dest = torch.rand((b, h, _n, _c), device='cuda', dtype=dtype)
        dout = torch.randn((b, h, r, n_ * c_, d, n_), device='cuda', dtype=dtype)
        ddenom = torch.randn((b, h, r, n_ * c_, n_), device='cuda', dtype=dtype)

        warm_up = 10
        for _ in range(warm_up):
            _, _, _, _, _ = _universal_attention_bwd(kc, vc, xq, static_src, static_dest, dout, ddenom)
            _, _, _, _, _ = universal_attention_backward(kc, vc, xq, static_src, static_dest, dout, ddenom)

        print("Checking running time...")
        n = 1000
        triton_time, torch_time = 0, 0
        for _ in range(n):
            kc = torch.rand((b, h, n_, c_, d), device='cuda', dtype=dtype)
            vc = torch.rand((b, h, n_, c_, d), device='cuda', dtype=dtype)
            xq = torch.rand((b, h, r, _n, _c, d), device='cuda', dtype=dtype)
            static_src = torch.rand((b, h, n_, c_), device='cuda', dtype=dtype)
            static_dest = torch.rand((b, h, _n, _c), device='cuda', dtype=dtype)
            dout = torch.rand((b, h, r, n_ * c_, d, n_), device='cuda', dtype=dtype)
            ddenom = torch.rand((b, h, r, n_ * c_, n_), device='cuda', dtype=dtype)

            start_time = time.time()
            _, _, _, _, _ = _universal_attention_bwd(kc, vc, xq, static_src, static_dest, dout, ddenom)
            triton_time += time.time() - start_time

            start_time = time.time()
            _, _, _, _, _ = universal_attention_backward(kc, vc, xq, static_src, static_dest, dout, ddenom)
            torch_time += time.time() - start_time

            del kc, vc, xq, static_src, static_dest, dout, ddenom

        print(f"Triton kernel time: {triton_time}s")
        print(f"Pytorch kernel time: {torch_time}s")

        print("Checking closeness to ground truth...")

        kc = torch.rand((b, h, n_, c_, d), device='cuda', dtype=dtype)
        vc = torch.rand((b, h, n_, c_, d), device='cuda', dtype=dtype)
        xq = torch.rand((b, h, r, _n, _c, d), device='cuda', dtype=dtype)
        static_src = torch.rand((b, h, n_, c_), device='cuda', dtype=dtype)
        static_dest = torch.rand((b, h, _n, _c), device='cuda', dtype=dtype)
        dout = torch.rand((b, h, r, n_ * c_, d, n_), device='cuda', dtype=dtype)
        ddenom = torch.rand((b, h, r, n_ * c_, n_), device='cuda', dtype=dtype)

        dkc, dvc, dxq, dstat_src, dstat_dest = universal_attention_backward_single_direction(kc, vc, xq, static_src, static_dest, dout, ddenom)
        dkc_ref, dvc_ref, dxq_ref, dstat_src_ref, dstat_dest_ref = universal_attention_backward(kc, vc, xq, static_src, static_dest, dout, ddenom)

        print("Checking dkc:")
        torch.testing.assert_close(dkc, dkc_ref, atol=1e-3, rtol=1e-5)
        print("Checking dvc:")
        torch.testing.assert_close(dvc, dvc_ref, atol=1e-3, rtol=1e-5)
        print("Checking dxq:")
        torch.testing.assert_close(dxq, dxq_ref, atol=1e-3, rtol=1e-5)
        print("Checking dstat_src:")
        torch.testing.assert_close(dstat_src, dstat_src_ref, atol=1e-3, rtol=1e-5)
        print("Checking dstat_dest:")
        torch.testing.assert_close(dstat_dest, dstat_dest_ref, atol=1e-3, rtol=1e-5)

    else:
        kc = torch.rand((b, h, n_, c_, d), device='cuda', dtype=dtype)
        vc = torch.rand((b, h, n_, c_, d), device='cuda', dtype=dtype)
        xq = torch.rand((b, h, r, _n, _c, d), device='cuda', dtype=dtype)
        static_src = torch.rand((b, h, n_, c_), device='cuda', dtype=dtype)
        static_dest = torch.rand((b, h, _n, _c), device='cuda', dtype=dtype)
        dout = torch.rand((b, h, r, n_ * c_, d, n_), device='cuda', dtype=dtype)
        ddenom = torch.rand((b, h, r, n_ * c_, n_), device='cuda', dtype=dtype)

        dkc, dvc, dxq, dstat_src, dstat_dest = _universal_attention_bwd(kc, vc, xq, static_src, static_dest, dout, ddenom)
        dkc_ref, dvc_ref, dxq_ref, dstat_src_ref, dstat_dest_ref = universal_attention_backward(kc, vc, xq, static_src, static_dest, dout, ddenom)
        _, _, _, _, _ = universal_attention_backward_single_direction(kc, vc, xq, static_src, static_dest, dout, ddenom)
        
        # count = 0
        # for arr in (dxq, dxq_ref):
        #     arr = arr.cpu().numpy().reshape(-1, d)
        #     np.savetxt(f"dxq{count}.csv", arr, delimiter=",", fmt="%.6f")   
        #     count += 1

        print("Checking dvc:")
        torch.testing.assert_close(dvc, dvc_ref, atol=1e-3, rtol=1e-5)
        print("Checking dxq:")
        torch.testing.assert_close(dxq, dxq_ref, atol=1e-3, rtol=1e-5)
        print("Checking dkc:")
        torch.testing.assert_close(dkc, dkc_ref, atol=1e-3, rtol=1e-5)
        print("Checking dstat_src:")
        torch.testing.assert_close(dstat_src, dstat_src_ref, atol=1e-3, rtol=1e-5)
        print("Checking dstat_dest:")
        torch.testing.assert_close(dstat_dest, dstat_dest_ref, atol=1e-3, rtol=1e-5)

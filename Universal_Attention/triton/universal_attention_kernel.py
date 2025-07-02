import torch
import triton
import triton.language as tl

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
    key=['s', 'd'],
)
@triton.jit
def _universal_attention_fwd_kernel(
    # Pointers to matrices
    q_ptr, k_ptr, v_ptr, src_ptr, dest_ptr, output_ptr, 
    spinlock_ptr, semaphore_ptr, sense_rev_ptr, 
    cum_cache_ptr, max_cache_ptr, sum_cache_ptr, 
    # Matrix dimensions
    b, n_kv, rep, s, d, 
    # Strides
    stride_q_b, stride_q_n_rep, stride_q_s, stride_q_d, 
    stride_k_b, stride_k_n_kv, stride_k_s, stride_k_d, 
    stride_v_b, stride_v_n_kv, stride_v_s, stride_v_d,
    stride_src_b, stride_src_n_kv, stride_src_s, 
    stride_dest_b, stride_dest_n_kv, stride_dest_s, 
    stride_output_b, stride_output_n_rep, stride_output_s, stride_output_d, 
    stride_spl_b, stride_spl_n_kv, stride_spl_s,
    stride_sem_b, stride_sem_n_kv, stride_sem_s,
    stride_rev_b, stride_rev_n_kv, stride_rev_s,
    stride_cum_b, stride_cum_n_kv, stride_cum_s,
    stride_max_b, stride_max_n_kv, stride_max_s, stride_max_c, 
    stride_sum_b, stride_sum_n_kv, stride_sum_s, stride_sum_c, 
    # Meta-parameters
    BLOCK_C: tl.constexpr, BLOCK_D: tl.constexpr, 
    C_BLOCK: tl.constexpr, D_BLOCK: tl.constexpr,
    DTYPE: tl.constexpr,
):
    pid_b_n_kv = tl.program_id(0)
    pid_b = pid_b_n_kv // n_kv
    pid_n_kv = pid_b_n_kv % n_kv
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    offs_m = pid_m * BLOCK_C + tl.arange(0, BLOCK_C)
    offs_n = pid_n * BLOCK_C + tl.arange(0, BLOCK_C)
    block_idx = tl.arange(0, C_BLOCK)

    affinity = tl.zeros((BLOCK_C, BLOCK_C), dtype=tl.float32)

    # k @ k^T
    k_ptr += pid_b * stride_k_b + pid_n_kv * stride_k_n_kv

    for d_offset in range(0, d, BLOCK_D):
        offs_d = d_offset + tl.arange(0, BLOCK_D)
        k_mat = tl.load(
            k_ptr + offs_m[:, None] * stride_k_s + offs_d[None, :] * stride_k_d, 
            mask=(offs_m[:, None] < s) & (offs_d[None, :] < d), 
            other=0.0
        )
        k_mat = tl.cast(k_mat, tl.float32)

        kt_mat = tl.load(
            k_ptr + offs_n[:, None] * stride_k_s + offs_d[None, :] * stride_k_d, 
            mask=(offs_n[:, None] < s) & (offs_d[None, :] < d), 
            other=0.0
        )
        kt_mat = tl.cast(kt_mat, tl.float32)

        # Use ieee to use fp32, otherwise the default would be tf32 even after tl.cast
        affinity += tl.dot(k_mat, tl.trans(kt_mat), input_precision="ieee")
    
    # .relu()
    affinity = tl.maximum(affinity, 0.0)

    # .pow(2/3)
    affinity = tl.exp2(tl.log2(affinity) * 2.0 / 3.0)

    # * static_src_.pow(1/3).unsqueeze(-1) * static_dest.pow(1/3).unsqueeze(-2)
    src_ptr += pid_b * stride_src_b + pid_n_kv * stride_src_n_kv
    dest_ptr += pid_b * stride_dest_b + pid_n_kv * stride_dest_n_kv

    src_mat  = tl.load(
        src_ptr + offs_m * stride_src_s, 
        mask=offs_m < s, 
        other=0.0
    )
    src_mat = tl.cast(src_mat, tl.float32)
    src_mat = tl.exp2(tl.log2(src_mat) / 3.0)

    dest_mat = tl.load(
        dest_ptr + offs_n * stride_dest_s, 
        mask=offs_n < s, 
        other=0.0
    )
    dest_mat = tl.cast(dest_mat, tl.float32)
    dest_mat = tl.exp2(tl.log2(dest_mat) / 3.0)

    affinity = affinity * src_mat[:, None] * dest_mat[None, :]

    # torch.log1p(affinity.clamp(min=0, max=1-1e-6).neg())
    affinity = tl.clamp(affinity, 0.0, 1.0 - 1e-6)
    affinity = tl.log(1.0 - affinity) 

    # .triu(1)
    # triu_mask = (offs_m[:, None] < offs_n[None, :])
    affinity = tl.where((offs_m[:, None] < offs_n[None, :]), affinity, 0.0)

    # .cumsum(3)
    spinlock_ptr += pid_b * stride_spl_b + pid_n_kv * stride_spl_n_kv + pid_m * stride_spl_s
    cum_cache_ptr += pid_b * stride_cum_b + pid_n_kv * stride_cum_n_kv 
    
    curr_sum = tl.sum(affinity, axis=1, keep_dims=False) # put the sum into the cache
    affinity = tl.cumsum(affinity, axis=1) # local cumsum

    # Make sure the sum is computed sequentially
    # while tl.load(spinlock_ptr, mask=True, other=0) < pid_n:
    while tl.atomic_add(spinlock_ptr, 0, sem="acquire") < pid_n:
        pass

    prev_sum = tl.zeros((BLOCK_C,), dtype=tl.float32)
    if pid_n > 0:
        prev_sum = tl.load(
            cum_cache_ptr + offs_m * stride_cum_s, 
            mask=offs_m < s, 
            other=0.0
        ) 
    curr_sum += prev_sum
    tl.store(
        cum_cache_ptr + offs_m * stride_cum_s, 
        curr_sum, 
        mask=offs_m < s
    )
    tl.atomic_add(spinlock_ptr, 1, sem="release")

    affinity += prev_sum[:, None]

    # .masked_fill(mask.tril(-1), -1e12)
    # tril_mask = (offs_m[:, None] > offs_n[None, :])
    affinity = tl.where((offs_m[:, None] > offs_n[None, :]), -1e12, affinity)

    # Affinity matrix completed here

    # k_.unsqueeze(2).matmul(xq.transpose(-1,-2)).add(affinity.unsqueeze(2))
    q_ptr += pid_b * stride_q_b + pid_n_kv * rep * stride_q_n_rep
    v_ptr += pid_b * stride_v_b + pid_n_kv * stride_v_n_kv
    output_ptr += pid_b * stride_output_b + pid_n_kv * rep * stride_output_n_rep

    semaphore_ptr += pid_b * stride_sem_b + pid_n_kv * stride_sem_n_kv + pid_m * stride_sem_s
    sense_rev_ptr += pid_b * stride_rev_b + pid_n_kv * stride_rev_n_kv + pid_m * stride_rev_s
    max_cache_ptr += pid_b * stride_max_b + pid_n_kv * stride_max_n_kv
    sum_cache_ptr += pid_b * stride_sum_b + pid_n_kv * stride_sum_n_kv
    
    # @Haochen: storing a 3D tensor after applying .dot() to 3D tensors would fail LLVM compilation
    # The problem is fixed in triton 3.2.0, and the alternative code is listed in matmul.py
    for r in range(0, rep):
        # k @ q^T
        output_rep_ptr = output_ptr + r * stride_output_n_rep
        kq = tl.zeros((BLOCK_C, BLOCK_C), dtype=tl.float32)

        for d_offset in range(0, d, BLOCK_D):
            offs_d = d_offset + tl.arange(0, BLOCK_D)

            k_mat = tl.load(
                k_ptr + offs_m[:, None] * stride_k_s + offs_d[None, :] * stride_k_d, 
                mask=(offs_m[:, None] < s) & (offs_d[None, :] < d), 
                other=0.0,
            )
            k_mat = tl.cast(k_mat, tl.float32)

            q_mat = tl.load(
                q_ptr + r * stride_q_n_rep + offs_n[:, None] * stride_q_s + offs_d[None, :] * stride_q_d, 
                mask=(offs_n[:, None] < s) & (offs_d[None, :] < d), 
                other=0.0,
            )
            q_mat = tl.cast(q_mat, tl.float32)

            # Use ieee to use fp32, otherwise the default would be tf32
            kq += tl.dot(k_mat, tl.trans(q_mat), input_precision="ieee")
        
        kq += affinity
        qk = tl.trans(kq)

        # Softmax(QK + mask)
        localmax = tl.max(qk, axis=1)
        tl.store(
            max_cache_ptr + offs_m * stride_max_s + pid_n * stride_max_c, 
            localmax, 
            mask=(offs_m < s)
        )

        qk = tl.exp(qk - localmax[:, None])
        qk_sumexp = tl.sum(qk, axis=1)
        tl.store(
            sum_cache_ptr + offs_m * stride_sum_s + pid_n * stride_sum_c, 
            qk_sumexp, 
            mask=(offs_m < s)
        )

        tl.atomic_add(semaphore_ptr, 1, sem="release")
        # Don't use atomic read here, or it will prevent other atomic operations
        # while tl.load(semaphore_ptr, mask=True, other=0) < C_BLOCK:
        while tl.atomic_add(semaphore_ptr, 0, sem="acquire") < C_BLOCK:
            pass

        localmax_mat = tl.load(
            max_cache_ptr + offs_m[:, None] * stride_max_s + block_idx[None, :] * stride_max_c,
            mask=(offs_m[:, None] < s) & (block_idx[None, :] < C_BLOCK),
            other=-1e9,
        )
        globalmax = tl.max(localmax_mat, axis=1)
        factor = tl.exp(localmax_mat - globalmax[:, None])

        sumexp_mat = tl.load(
            sum_cache_ptr + offs_m[:, None] * stride_sum_s + block_idx[None, :] * stride_sum_c,
            mask=(offs_m[:, None] < s) & (block_idx[None, :] < C_BLOCK),
            other=0.0,
        )
        globalsum = tl.sum(sumexp_mat * factor, axis=1)

        qk = qk * tl.exp(localmax - globalmax)[:, None]
        qk_softmax = tl.div_rn(qk, globalsum[:, None])

        # qk_softmax @ V
        if pid_n == 0:
            tl.atomic_xchg(semaphore_ptr, 0, sem="release")
        
        # while tl.load(semaphore_ptr, mask=True, other=0) > 0:
        while tl.atomic_add(semaphore_ptr, 0, sem="acquire") > 0:
            pass

        for idx in range(0, C_BLOCK + D_BLOCK - 1):
            # Clean the semaphore
            local_sense = tl.load(sense_rev_ptr, mask=True, other=0)
            num_proc = tl.atomic_add(semaphore_ptr, 1)
            if num_proc == C_BLOCK - 1:
                tl.atomic_xchg(semaphore_ptr, 0, sem="release")
                tl.atomic_xchg(sense_rev_ptr, 1 - local_sense, sem="release")
            else:
                while tl.atomic_add(sense_rev_ptr, 0, sem="acquire") == local_sense:
                # while tl.load(sense_rev_ptr, mask=True, other=0) == local_sense:
                    pass

            d_idx = idx - pid_n 
            if d_idx >= 0 and d_idx < D_BLOCK:
                offs_d = d_idx * BLOCK_D + tl.arange(0, BLOCK_D)
                v_mat = tl.load(
                    v_ptr + offs_n[:, None] * stride_v_s + offs_d[None, :] * stride_v_d,
                    mask=(offs_n[:, None] < s) & (offs_d[None, :] < d),
                    other=0.0,
                )
                prev_mat_sum = tl.load(
                    output_rep_ptr + offs_m[:, None] * stride_output_s + offs_d[None, :] * stride_output_d,
                    mask=(offs_m[:, None] < s) & (offs_d[None, :] < d),
                    other=0.0,
                )
                curr_mat_sum = prev_mat_sum + tl.dot(qk_softmax, v_mat, input_precision="ieee")
                # curr_mat_sum = tl.cast(curr_mat_sum, DTYPE) # Downcast to old datatype
                tl.store(
                    output_rep_ptr + offs_m[:, None] * stride_output_s + offs_d[None, :] * stride_output_d,
                    curr_mat_sum,
                    mask=(offs_m[:, None] < s) & (offs_d[None, :] < d),
                )

@triton.autotune(
    configs=configs,
    key=['r', 'n_', '_n', 'd'],
)
@triton.jit
def _universal_attention_fwd_kernel(
    # Pointers to matrices
    kc_ptr, vc_ptr, xq_ptr, kt_ptr, src_ptr, dest_ptr,                      # Input pointers
    out_ptr, denom_ptr, buffer_ptr,                                         # Output pointers
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
    IDX_J: tl.constexpr,                                                    # _n                                
    DTYPE: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_i = tl.program_id(2)              # n_
    offs_i = tl.arange(0, BLOCK_C)        # c_
    offs_j = tl.arange(0, BLOCK_R)        # _c
    offs_tri = pid_i * BLOCK_C - IDX_J * BLOCK_R
    offs_block = IDX_J * BLOCK_R + offs_j

    affinity = tl.zeros((BLOCK_C, BLOCK_R), dtype=tl.float32)

    # k_.matmul(_kt)
    kc_ptr += pid_b * str_kc_b + pid_h * str_kc_h + pid_i * str_kc_n_
    kt_ptr += pid_b * str_kt_b + pid_h * str_kt_h + IDX_J * str_kc__n

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
        affinity += tl.dot(k_mat, kt_mat, input_precision="ieee")

    # .relu()
    affinity = tl.maximum(affinity, 0.0)

    # .pow(2/3)
    affinity = tl.exp2(tl.log2(affinity) * 2.0 / 3.0)

    # * static_src_.pow(1/3).unsqueeze(-1) * static_dest.pow(1/3).unsqueeze(-2)
    src_ptr += pid_b * stride_src_b + pid_h * stride_src_h + pid_i * stride_src_n_ 
    dest_ptr += pid_b * stride_dest_b + pid_h * stride_dest_h + IDX_J * stride_dest__n

    src_mat  = tl.load(src_ptr + offs_i * stride_src_c_, mask=offs_i < BLOCK_C, other=0.0)
    src_mat = tl.cast(src_mat, tl.float32)
    src_mat = tl.exp2(tl.log2(src_mat) / 3.0)

    dest_mat = tl.load(dest_ptr + offs_j * stride_dest__c, mask=offs_j < BLOCK_R, other=0.0)
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
    buffer_ptr += pid_b * str_buffer_b + pid_h * str_buffer_h + pid_i * str_buffer_n_ 
    prev_sum = tl.load(buffer_ptr + offs_i * str_buffer_c_, mask=offs_i < BLOCK_C, other=0.0) 
    tl.store(buffer_ptr + offs_i * str_buffer_c_, curr_sum + prev_sum, mask=offs_i < BLOCK_C)

    # .cumsum(3)
    affinity = tl.cumsum(affinity, axis=1) + prev_sum[:, None]  

    # .masked_fill(mask.tril(i*c_-j*_c-1), -1e12)
    affinity = tl.where((offs_j[None, :] < (offs_i[:, None] + offs_tri)), -1e12, affinity)

    # k_.unsqueeze(2).matmul(_q.transpose(-1,-2)).add(affinity.unsqueeze(2))
    vc_ptr += pid_b * str_vc_b + pid_h * str_vc_h + pid_i * str_vc_n_
    xq_ptr += pid_b * str_xq_b + pid_h * str_xq_h + IDX_J * str_xq__n
    denom_ptr += pid_b * str_denom_b + pid_h * str_denom_h + pid_i * str_denom_n_ 
    denom_ptr = denom_ptr + offs_block * str_denom_l
    out_ptr += pid_b * str_out_b + pid_h * str_out_h + pid_i * str_out_n_
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
        score_exp = tl.exp(score_exp)

        score_sumexp = tl.sum(score_exp, axis=0, keep_dims=False)
        tl.store(denom_rep_ptr, tl.log(score_sumexp), mask=offs_block < BLOCK_R * _n)

        # score.transpose(-1,-2).softmax(dim=-1).to(dtype=_q.dtype).matmul(v_.unsqueeze(2))
        score_softmax = tl.div_rn(tl.trans(score_exp), score_sumexp[None, :])
        score_softmax = tl.cast(score_softmax, DTYPE) # b h r _c c_
        
        softmax_v = tl.zeros((BLOCK_R, BLOCK_C), dtype=tl.float32)
        for d_offset in range(0, d, BLOCK_D):
            vc_mat = tl.load(
                vc_ptr + offs_i[:, None] * str_vc_c_ + offs_d[None, :] * str_vc_d, 
                mask=(offs_i[:, None] < BLOCK_C) & (offs_d[None, :] < d), 
                other=0.0
            )
            # vc_mat = tl.cast(vc_mat, tl.float32)
            softmax_v += tl.dot(score_softmax, vc_mat, input_precision="ieee")

        tl.store(
            denom_rep_ptr[:, None] + offs_d[None, :] * str_out_d, 
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
    l = n_ * c_
    dtype = tl.float16 if xq.dtype == torch.float16 else tl.float32

    BLOCK_D = 64 # Placeholder for autotune

    # mask = torch.ones(c_,_c, dtype=torch.bool, device=xq.device)
    out = torch.empty(b,h,r,l,d,n_, dtype=xq.dtype, device=xq.device)
    denom = torch.empty(b,h,r,l,n_, dtype=xq.dtype, device=xq.device)
    sum_buffer = torch.zeros(b,h,n_,c_, dtype=static_src.dtype, device=static_src.device)

    kt = kc.view(b,h,_n,_c,d).permute(0,1,4,2,3)  # b h d _n _c

    # Due to the sum buffer for cumulative sum, we want to process that dimension sequencially
    grid = (b, h, n_)

    for j in range(_n):
        _universal_attention_fwd_kernel[grid](
            kc, vc, xq, kt, static_src, static_dest,                                
            out, denom, sum_buffer,                                                 
            b, h, r, n_, c_, _n, _c, d, 
            kc.stride(0), kc.stride(1), kc.stride(2), kc.stride(3), kc.stride(4),   
            vc.stride(0), vc.stride(1), vc.stride(2), vc.stride(3), vc.stride(4),   
            xq.stride(0), xq.stride(1), xq.stride(2), xq.stride(3), xq.stride(4), xq.stride(5),  
            kt.stride(0), kt.stride(1), kt.stride(2), kt.stride(3), kt.stride(4),   
            static_src.stride(0), static_src.stride(1), static_src.stride(2), static_src.stride(3), 
            static_dest.stride(0), static_dest.stride(1), static_dest.stride(2), static_dest.stride(3), 
            out.stride(0), out.stride(1), out.stride(2), out.stride(3), out.stride(4), out.stride(5), 
            denom.stride(0), denom.stride(1), denom.stride(2), denom.stride(3), denom.stride(4), 
            sum_buffer.stride(0), sum_buffer.stride(1), sum_buffer.stride(2), sum_buffer.stride(3),
            BLOCK_R=_c, BLOCK_C=c_, BLOCK_D=BLOCK_D, IDX_J=j, DTYPE=dtype,
        )
    return out, denom



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

            # b h d _n _c

            # Calculate decay matrix
            affinity = k_.matmul(_kt).relu().pow(2/3).float()  # deltanet style decay: b h c_ _c
            affinity = affinity * static_src_.pow(1/3).unsqueeze(-1) * _static_dest.pow(1/3).unsqueeze(-2)  # incorporate mamba-style and per-token decay
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
#################################
#     Ground Truth Autograd     #
#################################
'''
def universal_attention_forward(q, k, v, src, dest):
    b, h, l, d = k.shape
    _, nh, _, _ = q.shape
    c = 64
    n = l // c
    r = nh // h

    kc = k.view(b, h, n, c, d)
    vc = v.view(b, h, n, c, d)  
    xq = q.view(b, h, r, l, d)
    static_src = src.view(b, h, n, c)
    static_dest = dest.view(b, h, l)

    b,h,r,l,d = xq.shape
    _,_,n,c,_ = kc.shape

    mask = torch.ones(c,l, dtype=torch.bool, device=xq.device)
    out = torch.empty(b,h,r,l,d,n, dtype=xq.dtype, device=xq.device)
    denom = torch.empty(b,h,r,l,n, dtype=xq.dtype, device=xq.device)
    static_src = static_src.pow(1/3)
    static_dest = static_dest.pow(1/3)
    kt = kc.view(b,h,l,d).transpose(-2,-1)  # b h d l
    for i in range(n):
        k_ = kc[:,:,i]  # b h c d
        v_ = vc[:,:,i]  # b h c d
        static_src_ = static_src[:,:,i]  # b h c

        # Calculate decay matrix
        affinity = k_.matmul(kt).relu().pow(2/3).float()  # deltanet style decay
        affinity = affinity * static_src_.unsqueeze(-1) * static_dest.unsqueeze(-2)  # incorporate mamba-style and per-token decay
        affinity = torch.log1p(affinity.clamp(min=0, max=1-1e-6).neg())  # b h c l
        affinity = affinity.triu(i*c+1).cumsum(3)  # Accumulate decay with causal masking
        affinity = affinity.masked_fill(mask.tril(i*c-1), -1e12)  # Re-mask, with 1s on diagonal

        # Perform actual attention operation
        score = k_.unsqueeze(2).matmul(xq.transpose(-1,-2)).add(affinity.unsqueeze(2))  # b h r c l
        denom_ = score.logsumexp(dim=-2)  # b h r l
        out_ = score.transpose(-1,-2).softmax(dim=-1).to(dtype=xq.dtype).matmul(v_.unsqueeze(2))  # b h r l d

        out[...,i] = out_
        denom[...,i] = denom_
    output = out.mul(denom.softmax(dim=-1).unsqueeze(-2)).sum(-1)
    return output.reshape(b, nh, l, d)

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
    b, n_kv, rep, s, d = 2, 4, 4, 4096, 4096
    q = torch.rand((b, n_kv * rep, s, d), device='cuda', dtype=torch.float32)
    k = torch.rand((b, n_kv, s, d), device='cuda', dtype=torch.float32)
    v = torch.rand((b, n_kv, s, d), device='cuda', dtype=torch.float32)
    src = torch.rand((b, n_kv, s), device='cuda', dtype=torch.float32)
    dest = torch.rand((b, n_kv, s), device='cuda', dtype=torch.float32)

    output = _universal_attention_fwd(q, k, v, src, dest)
    output_ref = universal_attention_forward(q, k, v, src, dest)

    torch.testing.assert_close(output, output_ref)

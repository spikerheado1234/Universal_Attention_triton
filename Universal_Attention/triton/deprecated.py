## Here is a bunch of deprecated code. ##
@triton.jit
def _ua_fwd_kernel(
    # Pointers to Tensors
    Q, K, V, STATIC_SRC, STATIC_DEST, O, DENOM,
    # Strides
    s_q_b, s_q_h, s_q_n, s_q_d,
    s_k_b, s_k_h, s_k_n, s_k_d,
    s_v_b, s_v_h, s_v_n, s_v_d,
    s_ss_b, s_ss_h, s_ss_n,
    s_sd_b, s_sd_h, s_sd_n,
    s_o_b, s_o_h, s_o_n, s_o_d,
    s_d_b, s_d_h, s_d_n,
    # Shape and GQA Parameters
    KV_HEADS: tl.constexpr, HEAD_DIM: tl.constexpr, Q_H: tl.constexpr,
    # Chunking Parameters
    n_: tl.constexpr, c_: tl.constexpr, _n: tl.constexpr, _c: tl.constexpr,
    # Dtype
    DTYPE: tl.constexpr,
):
    i = tl.program_id(0)
    bhr_idx = tl.program_id(1)

    r_idx = bhr_idx % r
    h_idx = (bhr_idx // r) % h
    b_idx = bhr_idx // (h * r)

    ptr_k_ = KC + b_idx*s_kc_b + h_idx*s_kc_h + i*s_kc_n
    ptr_v_ = VC + b_idx*s_vc_b + h_idx*s_vc_h + i*s_vc_n
    ptr_ss_ = STATIC_SRC + b_idx*s_ss_b + h_idx*s_ss_h + i*s_ss_n
    
    offs_c_ = tl.arange(0, c_)
    offs_d = tl.arange(0, d)
    k_ = tl.load(ptr_k_ + offs_c_[:, None] * s_kc_c + offs_d[None, :] * s_kc_d)
    v_ = tl.load(ptr_v_ + offs_c_[:, None] * s_vc_c + offs_d[None, :] * s_vc_d)
    static_src_ = tl.load(ptr_ss_ + offs_c_ * s_ss_c)
    static_src = tl.exp2(tl.log2(static_src_.to(tl.float32)) / 3.0)

    sum_buffer = tl.zeros((c_,), dtype=tl.float32)
    offs__c = tl.arange(0, _c)
    
    for j in range(0, _n):
        ptr_q = XQ + b_idx*s_xq_b + h_idx*s_xq_h + r_idx*s_xq_r + j*s_xq_n
        l_idx = j * _c
        ptr_kt_j = KC + b_idx*s_kc_b + h_idx*s_kc_h + (l_idx // c_)*s_kc_n + (l_idx % c_)*s_kc_c
        ptr_sd = STATIC_DEST + b_idx*s_sd_b + h_idx*s_sd_h + j*s_sd_n

        q_ = tl.load(ptr_q + offs__c[:, None] * s_xq_c + offs_d[None, :] * s_xq_d)
        _kt_j = tl.load(ptr_kt_j + offs__c[:, None] * s_kc_c + offs_d[None, :] * s_kc_d)
        _static_dest = tl.load(ptr_sd + offs__c * s_sd_c)
        _static_dest = tl.exp2(tl.log2(_static_dest.to(tl.float32)) / 3)
        
        affinity = tl.dot(k_, tl.trans(_kt_j)).to(tl.float32)
        affinity = tl.where(affinity > 0, affinity, 0)
        affinity = tl.exp2(tl.log2(affinity) * 2.0/3.0)
        affinity = affinity * static_src_[:, None] * _static_dest[None, :]
        
        affinity = tl.where(affinity > 1.0 - 1e-6, 1.0 - 1e-6, affinity)
        affinity = tl.log1p(-affinity)
        
        global_k_indices = i * c_ + tl.arange(0, c_)
        global_q_indices = j * _c + tl.arange(0, _c)
        
        cumsum_mask = (global_k_indices[:, None] >= (global_q_indices[None, :] + 1))
        affinity = tl.where(cumsum_mask, affinity, 0.0)
        affinity = tl.cumsum(affinity, axis=1)
        
        affinity = affinity + sum_buffer[:, None]
        
        if _c > 0:
            last_col_offsets = (tl.arange(0, c_) * _c) + (_c - 1)
            sum_buffer = tl.load(affinity.ptr + last_col_offsets)

        remask = (global_k_indices[:, None] <= (global_q_indices[None, :] - 1))
        affinity = tl.where(remask, -1e12, affinity)

        score = tl.dot(k_, tl.trans(q_)).to(tl.float32)
        score = score + affinity

        m = tl.max(score, axis=0)
        lse = m + tl.log(tl.sum(tl.exp(score - m[None,:]), axis=0))
        
        probs = tl.exp(score - lse[None, :])
        _out = tl.dot(tl.trans(probs.to(DTYPE)), v_)

        ptr_o_base = O + b_idx*s_o_b + h_idx*s_o_h + r_idx*s_o_r + i*s_o_n
        ptr_o = ptr_o_base + (j*_c + offs__c[:, None]) * s_o_l + offs_d[None, :] * s_o_d
        ptr_denom_base = DENOM + b_idx*s_d_b + h_idx*s_d_h + r_idx*s_d_r + i*s_d_n
        ptr_denom = ptr_denom_base + (j*_c + offs__c) * s_d_l
        
        tl.store(ptr_o, _out)
        tl.store(ptr_denom, lse)

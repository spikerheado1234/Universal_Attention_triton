import torch
from torch.autograd import Function
import torch.nn.functional as F

import triton
import triton.language as tl

'''
######################################
#     Forward Kernel & Interface     #
######################################
'''
@triton.jit
def _aff_fwd_pre_cs(
    # Pointers to matrices
    k, src, dest, aff_pre_cs, 
    # Strides
    str_k_b, str_k_h, str_k_l, str_k_d,             # b h l d
    str_src_b, str_src_h, str_src_li,               # b h l
    str_dest_b, str_dest_h, str_dest_lj,            # b h l
    str_apc_b, str_apc_h, str_apc_li, str_apc_lj,   # b h l l
    # Matrix dimensions                                                 
    B: tl.constexpr, H: tl.constexpr, L: tl.constexpr, D: tl.constexpr,
    # Meta-parameters
    BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr
):
    pid_bh = tl.program_id(0)
    pid_b = pid_bh // H
    pid_h = pid_bh % H
    pid_i = tl.program_id(1)
    pid_j = tl.program_id(2)
    err = 1e-12

    k_ptr = k + pid_b * str_k_b + pid_h * str_k_h
    src_ptr = src + pid_b * str_src_b + pid_h * str_src_h
    dest_ptr = dest + pid_b * str_dest_b + pid_h * str_dest_h
    apc_ptr = aff_pre_cs + pid_b * str_apc_b + pid_h * str_apc_h

    offs_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    offs_j = pid_j * BLOCK_J + tl.arange(0, BLOCK_J)
    offs_d = tl.arange(0, D)

    src_vec = tl.load(
        src_ptr + offs_i * str_src_li,
        mask=(offs_i < L),
        other=0.0
    ).cast(tl.float32)
    src_mat = tl.sqrt_rn(src_vec + err)[:, None]

    dest_vec = tl.load(
        dest_ptr + offs_j * str_dest_lj,
        mask=(offs_j < L),
        other=0.0
    ).cast(tl.float32)
    dest_mat = tl.sqrt_rn(dest_vec + err)[:, None]

    k_mat = tl.load(
        k_ptr + offs_i[:, None] * str_k_l + offs_d[None, :] * str_k_d, 
        mask=(offs_i[:, None] < L) & (offs_d[None, :] < D), 
        other=0.0
    ).cast(tl.float32)

    kt_mat = tl.load(
        k_ptr + offs_j[:, None] * str_k_l + offs_d[None, :] * str_k_d, 
        mask=(offs_j[:, None] < L) & (offs_d[None, :] < D), 
        other=0.0
    ).cast(tl.float32)

    affinity = tl.dot(k_mat*src_mat, tl.trans(kt_mat*dest_mat), input_precision="ieee")

    # affinity = tl.exp2(tl.log2(tl.maximum(affinity, err)) * 2.0 / 3.0)
    # affinity = tl.log(1.0 - tl.clamp(affinity, 0.0, 1.0 - err)) 
    affinity = tl.where((affinity > 0), tl.exp2(tl.log2(affinity) * (2.0 / 3.0)), 0.0)
    affinity = tl.log(1.0 - tl.minimum(affinity, 1.0 - err))

    affinity = tl.where((offs_i[:, None] < offs_j[None, :]), affinity, 0.0)
    affinity = tl.cumsum(affinity, axis=1) # local cumsum

    tl.store(
        apc_ptr + offs_i[:, None] * str_apc_li + offs_j[None, :] * str_apc_lj, 
        affinity, 
        mask=(offs_i[:, None] < L) & (offs_j[None, :] < L)
    )

@triton.jit
def _aff_fwd_post_cs(
    # Pointers to matrices
    aff_pre_cs, global_sum, aff, 
    # Strides
    str_apc_b, str_apc_h, str_apc_li, str_apc_lj,   # b h l l
    str_gs_b, str_gs_h, str_gs_li, str_gs_j,        # b h l c
    str_aff_b, str_aff_h, str_aff_li, str_aff_lj,   # b h l l
    # Matrix dimensions                                                 
    B: tl.constexpr, H: tl.constexpr, L: tl.constexpr,
    # Meta-parameters
    BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr
):
    pid_bh = tl.program_id(0)
    pid_b = pid_bh // H
    pid_h = pid_bh % H
    pid_i = tl.program_id(1)
    pid_j = tl.program_id(2)

    apc_ptr = aff_pre_cs + pid_b * str_apc_b + pid_h * str_apc_h
    gs_ptr = global_sum + pid_b * str_gs_b + pid_h * str_gs_h
    aff_ptr = aff + pid_b * str_aff_b + pid_h * str_aff_h

    offs_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    offs_j = pid_j * BLOCK_J + tl.arange(0, BLOCK_J)

    apc_mat = tl.load(
        apc_ptr + offs_i[:, None] * str_apc_li + offs_j[None, :] * str_apc_lj, 
        mask=(offs_i[:, None] < L) & (offs_j[None, :] < L), 
        other=0.0
    ).cast(tl.float32)

    gs_vec = tl.load(
        gs_ptr + offs_i * str_gs_li + pid_j * str_gs_j, 
        mask=(offs_i < L), 
        other=0.0
    ).cast(tl.float32)

    affinity = apc_mat + gs_vec[:, None]
    affinity = tl.where((offs_i[:, None] > offs_j[None, :]), -1e12, affinity)

    tl.store(
        aff_ptr + offs_i[:, None] * str_aff_li + offs_j[None, :] * str_aff_lj, 
        affinity, 
        mask=(offs_i[:, None] < L) & (offs_j[None, :] < L)
    )

def _affinity_fwd(k, src, dest):
    '''
    Inputs:
    k:    b h l d
    src:  b h l
    dest: b h l

    Outputs:
    aff:  b h l l
    '''    
    b,h,l,d = k.shape
    assert src.shape == (b,h,l) and dest.shape == (b,h,l)
    BLOCK_I, BLOCK_J = 16, 16

    aff_pre_cs = torch.empty(b,h,l,l, dtype=torch.float32, device=k.device)
    aff = torch.empty(b,h,l,l, dtype=torch.float32, device=k.device)

    grid = (b*h, triton.cdiv(l, BLOCK_I), triton.cdiv(l, BLOCK_J))

    _aff_fwd_pre_cs[grid](
        k, src, dest, aff_pre_cs,                                                  
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        src.stride(0), src.stride(1), src.stride(2), 
        dest.stride(0), dest.stride(1), dest.stride(2), 
        aff_pre_cs.stride(0), aff_pre_cs.stride(1), aff_pre_cs.stride(2), aff_pre_cs.stride(3),
        B=b, H=h, L=l, D=d, 
        BLOCK_I=BLOCK_I, BLOCK_J=BLOCK_J
    )

    # Do cumsum in pytorch
    local_sum = aff_pre_cs[..., BLOCK_J-1:l:BLOCK_J]
    global_sum = torch.cumsum(F.pad(local_sum, (1, 0))[..., :-1], dim=-1).contiguous()

    _aff_fwd_post_cs[grid](
        aff_pre_cs, global_sum, aff,                                                 
        aff_pre_cs.stride(0), aff_pre_cs.stride(1), aff_pre_cs.stride(2), aff_pre_cs.stride(3),
        global_sum.stride(0), global_sum.stride(1), global_sum.stride(2), global_sum.stride(3),
        aff.stride(0), aff.stride(1), aff.stride(2), aff.stride(3),
        B=b, H=h, L=l, 
        BLOCK_I=BLOCK_I, BLOCK_J=BLOCK_J
    )

    return aff.transpose(-1, -2).to(k.dtype)

'''
#######################################
#     Backward Kernel & Interface     #
#######################################
'''
@triton.jit
def _aff_bwd_pre_cs_kernel(
    # Pointers to matrices
    daff, daff_cs,
    # Strides
    str_daff_b, str_daff_h, str_daff_li, str_daff_lj,   # b h l l
    str_dac_b, str_dac_h, str_dac_li, str_dac_lj,   # b h l l
    # Matrix dimensions
    B: tl.constexpr, H: tl.constexpr, L: tl.constexpr,
    # Meta-parameters
    BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr 
):
    pid_bh = tl.program_id(0)
    pid_b = pid_bh // H
    pid_h = pid_bh % H
    pid_i = tl.program_id(1)
    pid_j = tl.program_id(2)

    daff_ptr = daff + pid_b * str_daff_b + pid_h * str_daff_h
    dac_ptr = daff_cs + pid_b * str_dac_b + pid_h * str_dac_h

    offs_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    offs_j = pid_j * BLOCK_J + tl.arange(0, BLOCK_J)

    daff_mat = tl.load(
        daff_ptr + offs_j[:, None] * str_daff_lj + offs_i[None, :] * str_daff_li, 
        mask=(offs_j[:, None] < L) & (offs_i[None, :] < L), 
        other=0.0
    ).cast(tl.float32)

    daff_mat = tl.trans(daff_mat)
    daff_mat = tl.where((offs_i[:, None] > offs_j[None, :]), 0.0, daff_mat)
    daff_mat = tl.cumsum(daff_mat, axis=1, reverse=True) # local suffix sum

    tl.store(
        dac_ptr + offs_i[:, None] * str_dac_li + offs_j[None, :] * str_dac_lj, 
        daff_mat, 
        mask=(offs_i[:, None] < L) & (offs_j[None, :] < L)
    )

@triton.jit
def _aff_bwd_post_cs_kernel(
    # Pointers to matrices
    k, src, dest, daff_pre_cs, global_sum, daff_post_cs, daff_kkt,                                                                                             
    # Strides
    str_k_b, str_k_h, str_k_l, str_k_d,                     # b h l d
    str_src_b, str_src_h, str_src_li,                       # b h l
    str_dest_b, str_dest_h, str_dest_lj,                    # b h l
    str_daprc_b, str_daprc_h, str_daprc_li, str_daprc_lj,   # b h l l
    str_gs_b, str_gs_h, str_gs_li, str_gs_j,                # b h l c
    str_dapoc_b, str_dapoc_h, str_dapoc_li, str_dapoc_lj,   # b h l l
    str_dak_b, str_dak_h, str_dak_li, str_dak_lj,           # b h l l
    # Matrix dimensions                                         
    B: tl.constexpr, H: tl.constexpr, L: tl.constexpr, D: tl.constexpr, 
    # Meta-parameters
    BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr
):
    pid_bh = tl.program_id(0)
    pid_b = pid_bh // H
    pid_h = pid_bh % H
    pid_i = tl.program_id(1)
    pid_j = tl.program_id(2)
    err = 1e-12

    k_ptr = k + pid_b * str_k_b + pid_h * str_k_h
    src_ptr = src + pid_b * str_src_b + pid_h * str_src_h
    dest_ptr = dest + pid_b * str_dest_b + pid_h * str_dest_h
    daprc_ptr = daff_pre_cs + pid_b * str_daprc_b + pid_h * str_daprc_h
    gs_ptr = global_sum + pid_b * str_gs_b + pid_h * str_gs_h
    dapoc_ptr = daff_post_cs + pid_b * str_dapoc_b + pid_h * str_dapoc_h
    dak_ptr = daff_kkt + pid_b * str_dak_b + pid_h * str_dak_h

    offs_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    offs_j = pid_j * BLOCK_J + tl.arange(0, BLOCK_J)
    offs_d = tl.arange(0, D)

    src_vec = tl.load(
        src_ptr + offs_i * str_src_li,
        mask=(offs_i < L),
        other=0.0
    ).cast(tl.float32)
    src_mat = tl.sqrt_rn(src_vec + err)[:, None]

    dest_vec = tl.load(
        dest_ptr + offs_j * str_dest_lj,
        mask=(offs_j < L),
        other=0.0
    ).cast(tl.float32)
    dest_mat = tl.sqrt_rn(dest_vec + err)[:, None]

    k_mat = tl.load(
        k_ptr + offs_i[:, None] * str_k_l + offs_d[None, :] * str_k_d, 
        mask=(offs_i[:, None] < L) & (offs_d[None, :] < D), 
        other=0.0
    ).cast(tl.float32)

    kt_mat = tl.load(
        k_ptr + offs_j[:, None] * str_k_l + offs_d[None, :] * str_k_d, 
        mask=(offs_j[:, None] < L) & (offs_d[None, :] < D), 
        other=0.0
    ).cast(tl.float32)

    affinity1 = tl.dot(k_mat*src_mat, tl.trans(kt_mat*dest_mat), input_precision="ieee")

    # affinity2 = tl.exp2(tl.log2(tl.maximum(affinity1, err)) * 2.0 / 3.0)
    # affinity3 = tl.clamp(affinity2, 0.0, 1.0 - err)

    affinity2 = tl.where((affinity1 > 0), tl.exp2(tl.log2(affinity1) * (2.0 / 3.0)), 0.0)
    affinity3 = tl.minimum(affinity2, 1.0 - err)
  
    daprc_mat = tl.load(
        daprc_ptr + offs_i[:, None] * str_daprc_li + offs_j[None, :] * str_daprc_lj, 
        mask=(offs_i[:, None] < L) & (offs_j[None, :] < L), 
        other=0.0
    ).cast(tl.float32)

    gs_vec = tl.load(
        gs_ptr + offs_i * str_gs_li + pid_j * str_gs_j, 
        mask=(offs_i < L), 
        other=0.0
    ).cast(tl.float32)

    # daffinity = daprc_mat + gs_vec[:, None]
    # daffinity = tl.where((offs_i[:, None] > offs_j[None, :]), 0.0, daffinity)
    # daffinity = tl.div_rn(daffinity, 1.0-affinity4)
    # daffinity = tl.where((affinity3 > 0) & (affinity3 < 1.0 - err), -daffinity, 0.0)
    # daffinity = daffinity * (2.0/3.0) * tl.exp2(tl.log2(affinity2) * (-1.0/3.0))
    # daffinity = tl.where((affinity1 > 0), daffinity, 0.0)

    daffinity = daprc_mat + gs_vec[:, None]
    daffinity = tl.where((offs_i[:, None] > offs_j[None, :]), 0.0, daffinity)

    safe_inv = tl.where((affinity2 > 0) & (affinity2 < 1.0 - err), 1.0 / (1.0 - affinity3), 0.0)
    daffinity = -daffinity * safe_inv

    pow_grad = tl.where((affinity1 > 0), (2.0 / 3.0) * tl.exp2(tl.log2(affinity1) * (-1.0 / 3.0)), 0.0)
    daffinity = daffinity * pow_grad

    tl.store(
        dapoc_ptr + offs_i[:, None] * str_dapoc_li + offs_j[None, :] * str_dapoc_lj, 
        daffinity, 
        mask=(offs_i[:, None] < L) & (offs_j[None, :] < L)
    )

    tl.store(
        dak_ptr + offs_i[:, None] * str_dak_li + offs_j[None, :] * str_dak_lj, 
        daffinity * affinity1, 
        mask=(offs_i[:, None] < L) & (offs_j[None, :] < L)
    )

@triton.jit
def _aff_bwd_row_kernel(
    # Pointers to matrices
    k, src, dest, daff_post_cs, daff_kkt, dk_i, dsrc,
    # Strides
    str_k_b, str_k_h, str_k_l, str_k_d,                     # b h l d
    str_src_b, str_src_h, str_src_li,                       # b h l
    str_dest_b, str_dest_h, str_dest_lj,                    # b h l
    str_dapoc_b, str_dapoc_h, str_dapoc_li, str_dapoc_lj,   # b h l l
    str_dak_b, str_dak_h, str_dak_li, str_dak_lj,           # b h l l
    str_dki_b, str_dki_h, str_dki_l, str_dki_d,             # b h l d
    str_dsrc_b, str_dsrc_h, str_dsrc_li,                    # b h l
    # Matrix dimensions
    B: tl.constexpr, H: tl.constexpr, L: tl.constexpr, D: tl.constexpr,
    # Meta-parameters
    BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_i = tl.program_id(2)
    err = 1e-12

    k_ptr = k + pid_b * str_k_b + pid_h * str_k_h
    src_ptr = src + pid_b * str_src_b + pid_h * str_src_h
    dest_ptr = dest + pid_b * str_dest_b + pid_h * str_dest_h
    dapoc_ptr = daff_post_cs + pid_b * str_dapoc_b + pid_h * str_dapoc_h
    dak_ptr = daff_kkt + pid_b * str_dak_b + pid_h * str_dak_h
    dki_ptr = dk_i + pid_b * str_dki_b + pid_h * str_dki_h
    dsrc_ptr = dsrc + pid_b * str_dsrc_b + pid_h * str_dsrc_h

    offs_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    offs_d = tl.arange(0, D)

    src_vec = tl.load(
        src_ptr + offs_i * str_src_li,
        mask=(offs_i < L),
        other=0.0
    ).cast(tl.float32)
    src_mat = tl.sqrt_rn(src_vec + err)[:, None]

    dsrc_acc = tl.zeros((BLOCK_I,), dtype=tl.float32)
    dk_i_acc = tl.zeros((BLOCK_I, D), dtype=tl.float32)

    for j_offset in range(0, L, BLOCK_J):
        offs_j = j_offset + tl.arange(0, BLOCK_J)

        dak_mat = tl.load(
            dak_ptr + offs_i[:, None] * str_dak_li + offs_j[None, :] * str_dak_lj,
            mask=(offs_i[:, None] < L) & (offs_j[None, :] < L), 
            other=0.0
        ).to(tl.float32)
        dsrc_acc += tl.sum(dak_mat, axis=1)
        
        dapoc_mat = tl.load(
            dapoc_ptr + offs_i[:, None] * str_dapoc_li + offs_j[None, :] * str_dapoc_lj,
            mask=(offs_i[:, None] < L) & (offs_j[None, :] < L), other=0.0
        ).to(tl.float32)
        
        dest_vec = tl.load(
            dest_ptr + offs_j * str_dest_lj,
            mask=(offs_j < L),
            other=0.0
        ).cast(tl.float32)
        dest_mat = tl.sqrt_rn(dest_vec + err)[:, None]

        kt_mat = tl.load(
            k_ptr + offs_j[:, None] * str_k_l + offs_d[None, :] * str_k_d,
            mask=(offs_j[:, None] < L) & (offs_d[None, :] < D), 
            other=0.0
        ).to(tl.float32)
        
        dk_i_acc += tl.dot(dapoc_mat, kt_mat*dest_mat, input_precision="ieee")

    dsrc_vec = tl.div_rn(dsrc_acc, (2.0 * src_vec + 1e-12))
    tl.store(
        dsrc_ptr + offs_i * str_dsrc_li, 
        dsrc_vec, 
        mask=(offs_i < L)
    )

    dk_i_mat = dk_i_acc * src_mat
    tl.store(
        dki_ptr + offs_i[:, None] * str_dki_l + offs_d[None, :] * str_dki_d, 
        dk_i_mat, 
        mask=(offs_i[:, None] < L) & (offs_d[None, :] < D)
    )

@triton.jit
def _aff_bwd_col_kernel(
    # Pointers to matrices
    k, src, dest, daff_post_cs, daff_kkt, dk_j, ddest,
    # Strides
    str_k_b, str_k_h, str_k_l, str_k_d,                     # b h l d
    str_src_b, str_src_h, str_src_li,                       # b h l
    str_dest_b, str_dest_h, str_dest_lj,                    # b h l
    str_dapoc_b, str_dapoc_h, str_dapoc_li, str_dapoc_lj,   # b h l l
    str_dak_b, str_dak_h, str_dak_li, str_dak_lj,           # b h l l
    str_dkj_b, str_dkj_h, str_dkj_l, str_dkj_d,             # b h l d
    str_ddest_b, str_ddest_h, str_ddest_lj,                 # b h l
    # Matrix dimensions
    B: tl.constexpr, H: tl.constexpr, L: tl.constexpr, D: tl.constexpr,
    # Meta-parameters
    BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_j = tl.program_id(2)
    err = 1e-12
    
    k_ptr = k + pid_b * str_k_b + pid_h * str_k_h
    src_ptr = src + pid_b * str_src_b + pid_h * str_src_h
    dest_ptr = dest + pid_b * str_dest_b + pid_h * str_dest_h
    dapoc_ptr = daff_post_cs + pid_b * str_dapoc_b + pid_h * str_dapoc_h
    dak_ptr = daff_kkt + pid_b * str_dak_b + pid_h * str_dak_h
    dkj_ptr = dk_j + pid_b * str_dkj_b + pid_h * str_dkj_h
    ddest_ptr = ddest + pid_b * str_ddest_b + pid_h * str_ddest_h

    offs_j = pid_j * BLOCK_J + tl.arange(0, BLOCK_J)
    offs_d = tl.arange(0, D)

    dest_vec = tl.load(
        dest_ptr + offs_j * str_dest_lj,
        mask=(offs_j < L),
        other=0.0
    ).cast(tl.float32)
    dest_mat = tl.sqrt_rn(dest_vec + err)[:, None]

    ddest_acc = tl.zeros((BLOCK_J,), dtype=tl.float32)
    dk_j_acc = tl.zeros((BLOCK_J, D), dtype=tl.float32)

    for i_offset in range(0, L, BLOCK_I):
        offs_i = i_offset + tl.arange(0, BLOCK_I)

        dak_mat = tl.load(
            dak_ptr + offs_i[:, None] * str_dak_li + offs_j[None, :] * str_dak_lj,
            mask=(offs_i[:, None] < L) & (offs_j[None, :] < L), 
            other=0.0
        ).to(tl.float32)
        ddest_acc += tl.sum(dak_mat, axis=0)
        
        dapoc_mat = tl.load(
            dapoc_ptr + offs_i[:, None] * str_dapoc_li + offs_j[None, :] * str_dapoc_lj,
            mask=(offs_i[:, None] < L) & (offs_j[None, :] < L), other=0.0
        ).to(tl.float32)
        
        src_vec = tl.load(
            src_ptr + offs_i * str_src_li,
            mask=(offs_i < L),
            other=0.0
        ).cast(tl.float32)
        src_mat = tl.sqrt_rn(src_vec + err)[:, None]

        k_mat = tl.load(
            k_ptr + offs_i[:, None] * str_k_l + offs_d[None, :] * str_k_d,
            mask=(offs_i[:, None] < L) & (offs_d[None, :] < D), 
            other=0.0
        ).to(tl.float32)
        
        dk_j_acc += tl.dot(tl.trans(dapoc_mat), k_mat*src_mat, input_precision="ieee")

    ddest_vec = tl.div_rn(ddest_acc, (2.0 * dest_vec + 1e-12))
    tl.store(
        ddest_ptr + offs_j * str_ddest_lj, 
        ddest_vec, 
        mask=(offs_j < L)
    )

    dk_j_mat = dk_j_acc * dest_mat
    tl.store(
        dkj_ptr + offs_j[:, None] * str_dkj_l + offs_d[None, :] * str_dkj_d, 
        dk_j_mat, 
        mask=(offs_j[:, None] < L) & (offs_d[None, :] < D)
    )

def _affinity_bwd(k, src, dest, daff):
    '''
    Inputs:
    k:     b h l d
    src:   b h l
    dest:  b h l
    daff:  b h l l

    Outputs:
    dk:    b h l d
    dsrc:  b h l
    ddest: b h l
    '''    
    b,h,l,d = k.shape
    assert src.shape == (b,h,l) and dest.shape == (b,h,l) and daff.shape == (b,h,l,l)
    BLOCK_I, BLOCK_J = 16, 16

    daff_pre_cs = torch.empty(b,h,l,l, dtype=torch.float32, device=daff.device)
    daff_post_cs = torch.empty(b,h,l,l, dtype=torch.float32, device=daff.device)
    daff_kkt = torch.empty(b,h,l,l, dtype=torch.float32, device=daff.device)
    dk_i = torch.empty_like(k, dtype=torch.float32)
    dk_j = torch.empty_like(k, dtype=torch.float32)
    dsrc = torch.empty_like(src, dtype=torch.float32)
    ddest = torch.empty_like(dest, dtype=torch.float32)

    grid = (b*h, triton.cdiv(l, BLOCK_I), triton.cdiv(l, BLOCK_J))
    grid_i = (b, h, triton.cdiv(l, BLOCK_I))
    grid_j = (b, h, triton.cdiv(l, BLOCK_J))

    _aff_bwd_pre_cs_kernel[grid](
        daff, daff_pre_cs,
        daff.stride(0), daff.stride(1), daff.stride(2), daff.stride(3),
        daff_pre_cs.stride(0), daff_pre_cs.stride(1), daff_pre_cs.stride(2), daff_pre_cs.stride(3),
        B=b, H=h, L=l,
        BLOCK_I=BLOCK_I, BLOCK_J=BLOCK_J 
    )
    
    # Do cumsum in pytorch
    local_sum = torch.flip(daff_pre_cs[..., 0:l:BLOCK_J], dims=[-1])
    global_sum = torch.flip(torch.cumsum(F.pad(local_sum, (1, 0))[..., :-1], dim=-1), dims=[-1]).contiguous()
    
    local_sum = daff_pre_cs[..., 0:l:BLOCK_J]                 
    global_sum = torch.flip(torch.cumsum(torch.flip(local_sum, (-1,)), dim=-1), (-1,))
    global_sum = torch.roll(global_sum, shifts=-1, dims=-1)
    global_sum[..., -1] = 0

    _aff_bwd_post_cs_kernel[grid](
        k, src, dest, daff_pre_cs, global_sum, daff_post_cs, daff_kkt,
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        src.stride(0), src.stride(1), src.stride(2), 
        dest.stride(0), dest.stride(1), dest.stride(2),
        daff_pre_cs.stride(0), daff_pre_cs.stride(1), daff_pre_cs.stride(2), daff_pre_cs.stride(3), 
        global_sum.stride(0), global_sum.stride(1), global_sum.stride(2), global_sum.stride(3), 
        daff_post_cs.stride(0), daff_post_cs.stride(1), daff_post_cs.stride(2), daff_post_cs.stride(3), 
        daff_kkt.stride(0), daff_kkt.stride(1), daff_kkt.stride(2), daff_kkt.stride(3), 
        B=b, H=h, L=l, D=d,
        BLOCK_I=BLOCK_I, BLOCK_J=BLOCK_J 
    )

    # Two separate independent kernels for row-major and column-major reductions
    # s_i = torch.cuda.Stream()
    # s_j = torch.cuda.Stream()

    # with torch.cuda.stream(s_i):
    _aff_bwd_row_kernel[grid_i](
        k, src, dest, daff_post_cs, daff_kkt, dk_i, dsrc,
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        src.stride(0), src.stride(1), src.stride(2), 
        dest.stride(0), dest.stride(1), dest.stride(2),
        daff_post_cs.stride(0), daff_post_cs.stride(1), daff_post_cs.stride(2), daff_post_cs.stride(3), 
        daff_kkt.stride(0), daff_kkt.stride(1), daff_kkt.stride(2), daff_kkt.stride(3), 
        dk_i.stride(0), dk_i.stride(1), dk_i.stride(2), dk_i.stride(3),
        dsrc.stride(0), dsrc.stride(1), dsrc.stride(2), 
        B=b, H=h, L=l, D=d,
        BLOCK_I=BLOCK_I, BLOCK_J=BLOCK_J 
    )
    
    # with torch.cuda.stream(s_j):
    _aff_bwd_col_kernel[grid_j](
        k, src, dest, daff_post_cs, daff_kkt, dk_j, ddest,
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        src.stride(0), src.stride(1), src.stride(2), 
        dest.stride(0), dest.stride(1), dest.stride(2),
        daff_post_cs.stride(0), daff_post_cs.stride(1), daff_post_cs.stride(2), daff_post_cs.stride(3), 
        daff_kkt.stride(0), daff_kkt.stride(1), daff_kkt.stride(2), daff_kkt.stride(3), 
        dk_j.stride(0), dk_j.stride(1), dk_j.stride(2), dk_j.stride(3),
        ddest.stride(0), ddest.stride(1), ddest.stride(2), 
        B=b, H=h, L=l, D=d,
        BLOCK_I=BLOCK_I, BLOCK_J=BLOCK_J
    )

    # torch.cuda.synchronize()
    dk = dk_i + dk_j

    return dk.to(k.dtype), dsrc.to(src.dtype), ddest.to(dest.dtype)

class AffinityScores(Function):
    @staticmethod
    def forward(k, src, dest):
        aff = _affinity_fwd(k, src, dest)
        return aff
    
    @staticmethod
    def setup_context(ctx, inputs, outputs):
        k, src, dest = inputs
        # affinity = outputs
        ctx.save_for_backward(k, src, dest)
    
    @staticmethod
    def backward(ctx, daff):
        k, src, dest = ctx.saved_tensors
        daff = daff.contiguous()
        dk, dsrc, ddest = _affinity_bwd(k, src, dest, daff)
        return dk, dsrc, ddest

_gen_affinity_scores = AffinityScores.apply
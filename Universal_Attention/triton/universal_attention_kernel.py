import torch
import triton

# The pytorch autograd version is borrowed from here: 
# https://github.com/daviswer/torchtitan/blob/sandbox-selfprune-clean-wd/torchtitan/models/llama/utils.py

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=2),
    ],
    key=['hdim', 'dstate', 'chunk_size'],
)
@triton.jit
def _universal_attention_fwd_kernel(
    # Pointers to matrices
    # Matrix dimensions
    # Strides
    # Meta-parameters
):
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
    return out, denom

def _universal_attention_fwd(kc, vc, xq, static_src, static_dest):
    b, n_kv, rep, s, d = xq.shape
    _, _, n_c, c, _ = kc.shape
    device = xq.device
    assert kc.shape == (b, n_kv, n_c, c, d)
    assert vc.shape == (b, n_kv, n_c, c, d)
    assert static_src.shape == (b, n_kv, n_c, c)
    assert static_dest.shape == (b, n_kv, s)

    mask = torch.ones(c, s, dtype=torch.bool, device=device)
    out = torch.empty(b, n_kv, rep, s, d, n_c, dtype=xq.dtype, device=device)
    denom = torch.empty(b, n_kv, rep, s, n_c, dtype=xq.dtype, device=device)

    grid = lambda META: (triton.cdiv(headdim, META['BLOCK_SIZE_M']) * triton.cdiv(dstate, META['BLOCK_SIZE_N']),
                    batch * nchunks, nheads)
    
    with torch.cuda.device(x.device.index):
        _universal_attention_fwd_kernel[grid](
            x, B, states, dt, dA_cumsum, seq_idx,
            headdim, dstate, chunk_size,
            batch, seqlen, nheads // ngroups,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            B.stride(0), B.stride(1), B.stride(2), B.stride(-1),
            states.stride(0), states.stride(1), states.stride(2), states.stride(3), states.stride(4),
            dt.stride(0), dt.stride(2), dt.stride(1), dt.stride(3),
            dA_cumsum.stride(0), dA_cumsum.stride(2), dA_cumsum.stride(1), dA_cumsum.stride(3),
            *((seq_idx.stride(0), seq_idx.stride(1)) if seq_idx is not None else (0, 0)),
            HAS_SEQ_IDX=seq_idx is not None,
        )

    return out, denom

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=2),
    ],
    key=['hdim', 'dstate', 'chunk_size'],
)
@triton.jit
def _universal_attention_bwd_kernel(
    # Pointers to matrices
    # Matrix dimensions
    # Strides
    # Meta-parameters
):
    pass

def _universal_attention_bwd(kc, vc, xq, static_src, static_dest, dout, ddenom):

    dkc, dvc, dxq, dstatic_src, dstatic_dest = [torch.zeros_like(x) for x in [kc,vc,xq,static_src,static_dest]]
    # b,h,r,l,d = xq.shape
    # _,_,n,c,_ = kc.shape
    b, n_kv, rep, s, d = xq.shape
    _, _, n_c, c, _ = kc.shape
    mask = torch.ones(c,l, dtype=torch.bool, device=xq.device)
    static_src = static_src.pow(1/3)
    static_dest = static_dest.pow(1/3)
    kt = kc.view(b,h,l,d).transpose(-2,-1)  # b h d l

    for i in range(n):
        k_ = kc[:,:,i]  # b h c d
        v_ = vc[:,:,i]  # b h c d
        static_src_ = static_src[:,:,i]  # b h c
        dout_ = dout[...,i]
        ddenom_ = ddenom[...,i]

        # Rerun forward pass
        aff1 = k_.matmul(kt)
        aff2 = aff1.relu().pow(2/3).float()
        aff3 = aff2 * static_src_.unsqueeze(-1) * static_dest.unsqueeze(-2)
        score = torch.log1p(aff3.clamp(min=0,max=1-1e-6).neg()).triu(i*c+1).cumsum(3).masked_fill(mask.tril(i*c-1), -1e12)
        score = k_.unsqueeze(2).matmul(xq.transpose(-1,-2)).add(score.unsqueeze(2))  # b h r c l
        sscore = score.softmax(dim=-2)

        # Backward pass
        dvc[:,:,i] += sscore.to(dtype=dvc.dtype).matmul(dout_).sum(2)  # bhrcl,bhrld -> bhcd
        
        dscore = v_.unsqueeze(2).matmul(dout_.transpose(-1,-2))  # bhcd,bhrld -> bhrcl   <-- from out
        dscore = dscore.sub(dscore.mul(sscore).sum(-2,True)).mul(sscore)  # <-- from softmax
        dscore += score.softmax(-2) * ddenom_.unsqueeze(-2)  # b h r c l   <-- from denom

        dxq += dscore.to(dtype=dxq.dtype).transpose(-1,-2).matmul(k_.unsqueeze(2))  # bhrcl, bhcd -> bhrld
        dkc[:,:,i] += dscore.to(dtype=dkc.dtype).transpose(2,3).flatten(3,4).matmul(xq.flatten(2,3))  # bhrcl, bhrld -> bhcd

        daff = dscore.sum(2)  # b h c l
        daff = daff.flip([3]).cumsum(3).flip([3]).triu(i*c+1)  # <-- from cumsum
        daff /= aff3.clamp(min=1e-6, max=1-1e-6)-1  # <-- from ln(1-x)
        daff *= aff3.ge(0)
        daff *= aff3.le(1-1e-6)
        dstat = daff.mul(aff2).to(dtype=static_src.dtype)  # b h c l

        dstatic_src[:,:,i] += dstat.mul(static_dest.unsqueeze(-2)).sum(-1).div(static_src_.pow(2).mul(3))  # bhcl, bhl -> bhc
        dstatic_dest += dstat.mul(static_src_.unsqueeze(-1)).sum(-2).div(static_dest.pow(2).mul(3))  # bhcl, bhc -> bhl

        daff = daff.mul(static_src_.unsqueeze(-1)*static_dest.unsqueeze(-2))  # <-- from prod with statics
        daff = daff.to(dtype=xq.dtype) * aff1.abs().add(1e-9).pow(-1/3).mul(2/3).mul(aff1.gt(0))  # <-- from relu + pow

        dkc += daff.transpose(-1,-2).matmul(k_).view(b,h,n,c,d)  # bhcl, bhcd -> bhld, <-- grad via kt
        dkc[:,:,i] += daff.matmul(kt.transpose(-1,-2))  # bhcl, bhdl -> bhcd, <-- grad via k_

    return dkc, dvc, dxq, dstatic_src, dstatic_dest
    

import numpy as np
import inspect
import time

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
    # b, h, r, n_, c_, _n, _c, d = 2, 4, 2, 32, 64, 32, 64, 512
    b, h, r, n_, c_, _n, _c, d = 8, 4, 8, 128, 32, 128, 32, 128 # l = n * c = 4096

    # test = "forward"
    test = "backward"
    # test = "quick"

    if test == "forward":
        print("Testing forward pass")
        kc = torch.rand((b, h, n_, c_, d), device='cuda', dtype=torch.float32)
        vc = torch.rand((b, h, n_, c_, d), device='cuda', dtype=torch.float32)
        xq = torch.rand((b, h, r, _n, _c, d), device='cuda', dtype=torch.float32)
        static_src = torch.rand((b, h, n_, c_), device='cuda', dtype=torch.float32)
        static_dest = torch.rand((b, h, _n, _c), device='cuda', dtype=torch.float32)

        warm_up = 10
        for _ in range(warm_up):
            _, _ = _universal_attention_fwd(kc, vc, xq, static_src, static_dest)
            _, _ = universal_attention_forward(kc, vc, xq, static_src, static_dest)

        print("Checking running time...")
        n = 1000
        triton_time, torch_time = 0, 0
        for _ in range(n):
            kc = torch.rand((b, h, n_, c_, d), device='cuda', dtype=torch.float32)
            vc = torch.rand((b, h, n_, c_, d), device='cuda', dtype=torch.float32)
            xq = torch.rand((b, h, r, _n, _c, d), device='cuda', dtype=torch.float32)
            static_src = torch.rand((b, h, n_, c_), device='cuda', dtype=torch.float32)
            static_dest = torch.rand((b, h, _n, _c), device='cuda', dtype=torch.float32)

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

        print("Checking denom:")
        torch.testing.assert_close(denom, denom_ref, atol=1e-3, rtol=1e-5)
        print("Checking out:")
        torch.testing.assert_close(out, out_ref, atol=1e-3, rtol=1e-5)
        print("Checking output:")
        torch.testing.assert_close(output, output_ref, atol=1e-3, rtol=1e-5)
    
    elif test == "backward":
        print("Testing backward pass")
        kc = torch.rand((b, h, n_, c_, d), device='cuda', dtype=torch.float32)
        vc = torch.rand((b, h, n_, c_, d), device='cuda', dtype=torch.float32)
        xq = torch.rand((b, h, r, _n, _c, d), device='cuda', dtype=torch.float32)
        static_src = torch.rand((b, h, n_, c_), device='cuda', dtype=torch.float32)
        static_dest = torch.rand((b, h, _n, _c), device='cuda', dtype=torch.float32)
        dout = torch.randn((b, h, r, n_ * c_, d, n_), device='cuda', dtype=torch.float32)
        ddenom = torch.randn((b, h, r, n_ * c_, n_), device='cuda', dtype=torch.float32)

        warm_up = 10
        for _ in range(warm_up):
            _, _, _, _, _ = _universal_attention_bwd(kc, vc, xq, static_src, static_dest, dout, ddenom)
            _, _, _, _, _ = universal_attention_backward(kc, vc, xq, static_src, static_dest, dout, ddenom)

        print("Checking running time...")
        n = 1000
        triton_time, torch_time = 0, 0
        for _ in range(n):
            kc = torch.rand((b, h, n_, c_, d), device='cuda', dtype=torch.float32)
            vc = torch.rand((b, h, n_, c_, d), device='cuda', dtype=torch.float32)
            xq = torch.rand((b, h, r, _n, _c, d), device='cuda', dtype=torch.float32)
            static_src = torch.rand((b, h, n_, c_), device='cuda', dtype=torch.float32)
            static_dest = torch.rand((b, h, _n, _c), device='cuda', dtype=torch.float32)
            dout = torch.rand((b, h, r, n_ * c_, d, n_), device='cuda', dtype=torch.float32)
            ddenom = torch.rand((b, h, r, n_ * c_, n_), device='cuda', dtype=torch.float32)

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

        kc = torch.rand((b, h, n_, c_, d), device='cuda', dtype=torch.float32)
        vc = torch.rand((b, h, n_, c_, d), device='cuda', dtype=torch.float32)
        xq = torch.rand((b, h, r, _n, _c, d), device='cuda', dtype=torch.float32)
        static_src = torch.rand((b, h, n_, c_), device='cuda', dtype=torch.float32)
        static_dest = torch.rand((b, h, _n, _c), device='cuda', dtype=torch.float32)
        dout = torch.rand((b, h, r, n_ * c_, d, n_), device='cuda', dtype=torch.float32)
        ddenom = torch.rand((b, h, r, n_ * c_, n_), device='cuda', dtype=torch.float32)

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
        kc = torch.rand((b, h, n_, c_, d), device='cuda', dtype=torch.float32)
        vc = torch.rand((b, h, n_, c_, d), device='cuda', dtype=torch.float32)
        xq = torch.rand((b, h, r, _n, _c, d), device='cuda', dtype=torch.float32)
        static_src = torch.rand((b, h, n_, c_), device='cuda', dtype=torch.float32)
        static_dest = torch.rand((b, h, _n, _c), device='cuda', dtype=torch.float32)
        dout = torch.rand((b, h, r, n_ * c_, d, n_), device='cuda', dtype=torch.float32)
        ddenom = torch.rand((b, h, r, n_ * c_, n_), device='cuda', dtype=torch.float32)

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

import torch
from torch.autograd import Function
from Universal_Attention.triton.universal_attention_kernel import _universal_attention_fwd, _universal_attention_bwd

# The pytorch autograd version is borrowed from here: 
# https://github.com/daviswer/torchtitan/blob/sandbox-selfprune-clean-wd/torchtitan/models/llama/utils.py

class UniversalAttention(Function):
    @staticmethod
    def forward(kc, vc, xq, static_src, static_dest):
        b, n_kv, rep, s, d = xq.shape
        _, _, n_c, c, _ = kc.shape
        assert kc.shape == (b, n_kv, n_c, c, d)
        assert vc.shape == (b, n_kv, n_c, c, d)
        assert static_src.shape == (b, n_kv, n_c, c)
        assert static_dest.shape == (b, n_kv, s)
        # TODO: Set contiguous tensors
        out, denom = _universal_attention_fwd(kc, vc, xq, static_src, static_dest)
        return out, denom

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        kc, vc, xq, static_src, static_dest = inputs
        # out, denom = outputs
        ctx.save_for_backward(kc, vc, xq, static_src, static_dest)

    @staticmethod
    def backward(ctx, dout, ddenom):
        # Note: when using mixed precision, dout is downcast but ddenom is always fp32
        kc, vc, xq, static_src, static_dest = ctx.saved_tensors
        b, n_kv, rep, s, d = xq.shape
        _, _, n_c, c, _ = kc.shape
        print(dout.shape, ddenom.shape)
        # assert dout.shape == (b)
        # assert ddenom.shape == (b)
        # TODO: Set contiguous tensors
        dkc, dvc, dxq, dstatic_src, dstatic_dest = _universal_attention_bwd(kc, vc, xq, static_src, static_dest, dout, ddenom)
        # TODO: Handle datatype
        return dkc, dvc, dxq, dstatic_src, dstatic_dest

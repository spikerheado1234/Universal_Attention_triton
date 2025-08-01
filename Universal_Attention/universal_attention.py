import torch
from torch.autograd import Function
from Universal_Attention.triton.universal_attention_kernel import _universal_attention_fwd, _universal_attention_bwd

class UniversalAttention(Function):
    @staticmethod
    def forward(kc, vc, xq, static_src, static_dest):
        b, h, r, _n, _c, d = xq.shape
        _, _, n_, c_, _ = kc.shape
        assert kc.shape == (b, h, n_, c_, d)
        assert vc.shape == (b, h, n_, c_, d)
        assert static_src.shape == (b, h, n_, c_)
        assert static_dest.shape == (b, h, _n, _c)
        if kc.stride(-1) != 1:
            kc = kc.contiguous()
        if vc.stride(-1) != 1:
            vc = vc.contiguous()
        if xq.stride(-1) != 1:
            xq = xq.contiguous()
        if static_src.stride(-1) != 1:
            static_src = static_src.contiguous()
        if static_dest.stride(-1) != 1:
            static_dest = static_dest.contiguous()
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
        b, h, r, _n, _c, d = xq.shape
        _, _, n_, c_, _ = kc.shape
        assert dout.shape == (b, h, r, n_ * c_, d, n_)
        assert ddenom.shape == (b, h, r, n_ * c_, n_)
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        if ddenom.stride(-1) != 1:
            ddenom = ddenom.contiguous()
        dkc, dvc, dxq, dstatic_src, dstatic_dest = _universal_attention_bwd(kc, vc, xq, static_src, static_dest, dout, ddenom)
        return dkc, dvc, dxq, dstatic_src, dstatic_dest

# class UniversalAttention(Function):
#     @staticmethod
#     def forward(q, k, v, src, dest):
#         b, n, s, d = q.shape
#         _, n_kv, _, _ = k.shape
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
#         # TODO: Handle datatype/precision conversion
#         return dq, dk, dv, dsrc, ddest

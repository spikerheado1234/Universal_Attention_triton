## File that has some debugging helpe functions. ##
import torch
from universal_attention_kernel_opt import attention

def sdpa_torch(q, k, v):
    """
    Einsum variant of sdpa, to track intermediate tensor values.
    """
    qk = torch.einsum('bnqh, bnkh -> bnqk', q, k)/torch.sqrt(q.shape[-1])
    print(f'qk: {qk}')
    attn = torch.nn.functional.softmax(qk, dim=-1)
    print(f'attn: {attn}')
    o = torch.einsum('bnqk, bnkh -> bnqh', attn, v)
    print(f'o: {o}')
    return o

def _debug_triton_fused_mhsa(q,k,v):
    sdpa_torch(q,k,v)
    sm_scale = 1.3
    causal=False
    fn = lambda: attention(q, k, v, causal, sm_scale)
    fn()


if __name__ == '__main__':
    ## Here we call whatever we would like to debug. ##
    BATCH=1
    H=2
    N_CTX=128
    HEAD_DIM=32
    device="cuda" if torch.cuda.is_available() else "cpu"
    provider = "triton" ## triton/flash.
    dtype=torch.float16
    q = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    k = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    v = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    _debug_triton_fused_mhsa(q,k,v)


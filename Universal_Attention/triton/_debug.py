## File that has some debugging helpe functions. ##
import torch
from universal_attention_kernel_opt import attention
from math import sqrt

def sdpa_torch(q, k, v):
    """
    Einsum variant of sdpa, to track intermediate tensor values.
    """
    print(f'q[:16] = {q[:,:,:16,:].sum()}')
    print(f'q[16:] = {q[:,:,16:,:].sum()}')
    print(f'k[:16] = {k[:, :, :16,:].sum()}')
    print(f'k[16:] = {k[:, :, 16:,:].sum()}')
    qk = torch.einsum('bnqh, bnkh -> bnqk', q, k)/torch.full((q.shape[0],q.shape[1],q.shape[2],k.shape[2]), sqrt(q.shape[-1])).to(q.device)
    print(f'qk[:16]: {qk[:, :, :16, :].sum()}')
    print(f'qk[16:]: {qk[:, :, 16:, :].sum()}')
    attn = torch.nn.functional.softmax(qk, dim=-1,dtype=torch.float32)
    #print(f'attn: {attn}')
    o = torch.einsum('bnqk, bnkh -> bnqh', attn.to(dtype=q.dtype), v)
    #print(f'o: {o}')

    ## Here are some additional intermediate tensors we print for further sanity checking. ##
    print(f'exp(qk)[:16,:16], {torch.exp(qk)[:,:,:16,:16].sum()}')
    return o

def _debug_triton_fused_mhsa(q,k,v):
    sdpa_output = sdpa_torch(q,k,v)
    sm_scale = 1.3
    causal=False
    fn = lambda: attention(q, k, v, causal, sm_scale)
    triton_output = fn()
    print(f'sdpa_output: {sdpa_output}')
    print(f'triton_output: {triton_output}')
    print(f'allclose: {torch.allclose(sdpa_output, triton_output, atol=1e-1, rtol=1e-1)}')


if __name__ == '__main__':
    ## Here we call whatever we would like to debug. We instantiate with debug sized tensors. ##
    torch.manual_seed(0)
    BATCH=1
    H=1
    N_CTX=32
    HEAD_DIM=16
    device="cuda" if torch.cuda.is_available() else "cpu"
    provider = "triton" ## triton/flash.
    dtype=torch.float16
    q = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    k = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    v = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    _debug_triton_fused_mhsa(q,k,v)


## File that has some debugging helpe functions. ##
import torch
from universal_attention_kernel_opt import attention
from math import sqrt

def sdpa_torch(q, k, v):
    """
    Einsum variant of sdpa, to track intermediate tensor values.
    """
    qk = torch.einsum('bnqh, bnkh -> bnqk', q, k)/torch.full((q.shape[0],q.shape[1],q.shape[2],k.shape[2]), sqrt(q.shape[-1])).to(q.device)
    attn = torch.nn.functional.softmax(qk, dim=-1,dtype=torch.float32)
    o = torch.einsum('bnqk, bnkh -> bnqh', attn.to(dtype=q.dtype), v)
    return o

def _debug_triton_fused_mhsa(q,k,v, backward=False):
    ## We clone and detach the tensors for grad checking. ##
    q_torch = q.clone().detach().requires_grad_(True)
    k_torch = k.clone().detach().requires_grad_(True)
    v_torch = v.clone().detach().requires_grad_(True)
    sdpa_output = sdpa_torch(q_torch,k_torch,v_torch)
    sm_scale = 1.3
    causal=False
    fn = lambda: attention(q, k, v, causal, sm_scale)
    do = torch.randn_like(q).to(q.device)
    triton_output = fn()
    if backward:
        fn = lambda: triton_output.backward(do, retain_graph=True)
        fn()
        sdpa_output.backward(do, retain_graph=True)

    print(f'sdpa_output: {sdpa_output}')
    print(f'triton_output: {triton_output}')
    print(f'outputs allclose: {torch.allclose(sdpa_output, triton_output, atol=1e-1, rtol=1e-1)}')
    if backward:
        print(f'dq allclose: {torch.allclose(q_torch.grad, q.grad, atol=1e-1, rtol=1e-1)}')
        print(f'dk allclose: {torch.allclose(k_torch.grad, k.grad, atol=1e-1, rtol=1e-1)}')
        print(f'dv allclose: {torch.allclose(v_torch.grad, v.grad, atol=1e-1, rtol=1e-1)}')

if __name__ == '__main__':
    ## Here we call whatever we would like to debug. We instantiate with debug sized tensors. ##
    torch.manual_seed(0)
    BATCH=1
    H=1
    N_CTX=128
    HEAD_DIM=16
    device="cuda" if torch.cuda.is_available() else "cpu"
    provider = "triton" ## triton/flash.
    dtype=torch.float16
    q = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    k = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    v = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    _debug_triton_fused_mhsa(q,k,v, backward=True)


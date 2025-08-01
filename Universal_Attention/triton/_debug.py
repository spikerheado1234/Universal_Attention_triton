## File that has some debugging helpe functions. ##
import torch
from universal_attention_kernel_opt import attention
from math import sqrt
import pdb

def sdpa_torch(q, k, v, causal=False):
    """
    Einsum variant of sdpa, to track intermediate tensor values.
    """
    qk = torch.einsum('bnqh, bnkh -> bnqk', q, k)/torch.full((q.shape[0],q.shape[1],q.shape[2],k.shape[2]), sqrt(q.shape[-1])).to(q.device) ## S
    if causal:
        qk += torch.triu(torch.full(qk.shape, -1e6), diagonal=1).to(q.device)
    attn = torch.nn.functional.softmax(qk, dim=-1,dtype=torch.float32) ## P
    o = torch.einsum('bnqk, bnkh -> bnqh', attn.to(dtype=q.dtype), v) ## O
    return o, attn.to(dtype=q.dtype)

def sdpa_torch_bwd(attn, k, v, q, o, incoming_gradients):
    ## Compute the gradient of v & attn ##
    print(f'q: {q.sum()}')
    print(f'k: {k.sum()}')
    print(f'v: {v.sum()}')
    dv = torch.einsum('bnqk, bnqh -> bnkh', attn, incoming_gradients)
    dattn = torch.einsum('bnqh, bnkh -> bnqk', incoming_gradients, v)

    ## next, we compute the derivative of qk. We use the trick that: dqk = attn * dattn - row_sum(incoming_gradients * o) * attn. ##
    ##  here d = row_sum(incoming_gradients * o).
    d = torch.sum(incoming_gradients * o, axis=-1)  ## (b, n, s) -> Similar sized shape to logsumexp.
    print(f'd: {d.sum()}')
    dqk = (attn * dattn) - (torch.unsqueeze(d, dim=-1) * attn)  ## Check the correctness of this, may be incorrect.
    print(f'dqk: {dqk.sum()}')
    print(f'attn: {attn.sum()}')

    ## Finally, we compute dq and dk. ##
    dq = torch.einsum('bnqk, bnkt -> bnqt', dqk, k) / torch.full(q.shape, sqrt(q.shape[-1])).to(q.device).to(dtype=q.dtype)
    dk = torch.einsum('bnkq, bnkt -> bnqt', dqk, q) / torch.full(k.shape, sqrt(q.shape[-1])).to(q.device).to(dtype=q.dtype)

    return dq, dk, dv 

def _debug_triton_fused_mhsa(q,k,v, backward=False, causal=False):
    ## We clone and detach the tensors for grad checking. ##
    q_torch = q.clone().detach().requires_grad_(True)
    k_torch = k.clone().detach().requires_grad_(True)
    v_torch = v.clone().detach().requires_grad_(True)
    sdpa_output, attn = sdpa_torch(q_torch, k_torch, v_torch, causal)
    q_sanity = q.clone().detach().requires_grad_(True)
    k_sanity = k.clone().detach().requires_grad_(True)
    v_sanity = v.clone().detach().requires_grad_(True)
    sm_scale = 1.3
    fn = lambda: attention(q, k, v, causal, sm_scale)
    do = torch.randn_like(q).to(q.device)
    triton_output = fn()
    if backward:
        dq_sanity, dk_sanity, dv_sanity = sdpa_torch_bwd(attn, k_sanity, v_sanity, q_sanity, sdpa_output, do)
        fn = lambda: triton_output.backward(do, retain_graph=True)
        fn()
        sdpa_output.backward(do, retain_graph=True)

    print(f'outputs allclose: {torch.allclose(sdpa_output, triton_output, atol=1e-1, rtol=1e-1)}')
    if backward:
        #print(f'q_torch.grad: {q_torch.grad}')
        #print(f'q.grad: {q.grad}')
        print(f'dq allclose: {torch.allclose(q_torch.grad, q.grad, atol=1e-1, rtol=1e-1)}')
        print(f'dk allclose: {torch.allclose(k_torch.grad, k.grad, atol=1e-1, rtol=1e-1)}')
        print(f'dv allclose: {torch.allclose(v_torch.grad, v.grad, atol=1e-1, rtol=1e-1)}')
        print(f'------output sanity checking------')
        print(f'dq allclose: {torch.allclose(q_torch.grad, dq_sanity, atol=1e-1, rtol=1e-1)}')
        print(f'dk allclose: {torch.allclose(k_torch.grad, dk_sanity, atol=1e-1, rtol=1e-1)}')
        print(f'dv allclose: {torch.allclose(v_torch.grad, dv_sanity, atol=1e-1, rtol=1e-1)}')


if __name__ == '__main__':
    ## Here we call whatever we would like to debug. We instantiate with debug sized tensors. ##
    torch.manual_seed(0)
    BATCH=1
    H=1
    N_CTX=16
    HEAD_DIM=16
    causal = True
    device="cuda" if torch.cuda.is_available() else "cpu"
    provider = "triton" ## triton/flash.
    dtype=torch.float16
    q = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    k = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    v = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    _debug_triton_fused_mhsa(q,k,v, backward=True, causal=causal)


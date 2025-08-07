## File that has some debugging helpe functions. ##
import torch
from universal_attention_kernel_opt import attention, _gen_affinity_scores
from math import sqrt
import pdb
from universal_attention_kernel import universal_attention_forward
from math import ceil
import time

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

def sdpa_gqa_torch(q, k, v, causal=False):
    """
    Einsum variant of sdpa-gqa, to track intermediate tensor values for easier debugging.

    h: number of kv groups.
    r: number of queries shared per kv group.
    q.shape[1] % k.shape[1] == 0.
    h = k.shape[1]
    r = q.shape[1] // k.shape.
    """
    assert q.shape[1] % k.shape[1] == 0, 'Incorrect number of heads for gqa passed in.'
    incoming_query_shape = q.shape
    q = torch.reshape(q, shape=(q.shape[0], q.shape[1] // k.shape[1], k.shape[1], q.shape[2], q.shape[3]))
    qk = torch.einsum('brnqh, bnkh -> brnqk', q, k)/torch.full((q.shape[0], q.shape[1], q.shape[2], q.shape[3], k.shape[2]), sqrt(q.shape[-1])).to(q.device) ## S
    if causal:
        qk += torch.triu(torch.full(qk.shape, -1e6), diagonal=1).to(q.device)
    attn = torch.nn.functional.softmax(qk, dim=-1,dtype=torch.float32) ## P
    o = torch.einsum('brnqk, bnkh -> brnqh', attn.to(dtype=q.dtype), v) ## O
    return torch.reshape(o, shape=incoming_query_shape), attn.to(dtype=q.dtype)

def sdpa_gqa_torch_bwd(attn, k, v, q, o, incoming_gradients):
    ## Compute the gradient of v & attn ##
    assert q.shape[1] % k.shape[1] == 0, 'Incorrect number of heads for gqa passed in.'
    incoming_gradients = torch.reshape(incoming_gradients, (q.shape[0], q.shape[1] // k.shape[1], k.shape[1], q.shape[2], q.shape[3]))
    q = torch.reshape(q, shape=(q.shape[0], q.shape[1] // k.shape[1], k.shape[1], q.shape[2], q.shape[3]))
    o = torch.reshape(o, incoming_gradients.shape)
    dv = torch.einsum('brnqk, brnqh -> bnkh', attn, incoming_gradients)
    dattn = torch.einsum('brnqh, bnkh -> brnqk', incoming_gradients, v)

    ## next, we compute the derivative of qk. We use the trick that: dqk = attn * dattn - row_sum(incoming_gradients * o) * attn. ##
    ##  here d = row_sum(incoming_gradients * o).
    d = torch.sum(incoming_gradients * o, axis=-1)  ## (b, r, n, s) -> Similar sized shape to logsumexp.
    dqk = (attn * dattn) - (torch.unsqueeze(d, dim=-1) * attn)  ## Check the correctness of this, may be incorrect.

    ## Finally, we compute dq and dk. ##
    dq = torch.einsum('brnqk, bnkt -> brnqt', dqk, k) / torch.full(incoming_gradients.shape, sqrt(q.shape[-1])).to(q.device).to(dtype=q.dtype)
    dk = torch.einsum('brnkq, brnkt -> bnqt', dqk, q) / torch.full(k.shape, sqrt(q.shape[-1])).to(q.device).to(dtype=q.dtype)

    return torch.reshape(dq, (dq.shape[0], dq.shape[1] * dq.shape[2], dq.shape[3], dq.shape[4])), dk, dv 


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

## Currently untested. TODO(ahangupta): test this function. ##
def ua_bwd(q, k, v, src, dest, incoming_gradients):
    assert q.shape[1] % k.shape[1] == 0, 'Incorrect number of heads for gqa passed in.'
    incoming_query_shape = q.shape
    q = torch.reshape(q, shape=(q.shape[0], q.shape[1] // k.shape[1], k.shape[1], q.shape[2], q.shape[3]))
    incoming_gradients = torch.reshape(incoming_gradients, shape=q.shape)
    qk = torch.einsum('brnqh, bnkh -> brnqk', q, k)

    affinity = _gen_affinity_scores(k, src, dest)
    qk += affinity.unsqueeze(1)
    attn = torch.nn.functional.softmax(qk, dim=-1, dtype=torch.float32)
    attn = attn.to(q.dtype)
    o = torch.einsum('brnqk, bnkh -> brnqh', attn.to(dtype=q.dtype), v)

    ## Now, we compute the gradients. ##

    ## dv. ##
    dv = torch.einsum('brnqk, brnqh -> bnkh', attn, incoming_gradients)

    ## dq/dk. ##
    dattn = torch.einsum('brnqh, bnkh -> brnqk', incoming_gradients, v)
    d = torch.sum(incoming_gradients * o, axis=-1)  ## (b, n, s) -> Similar sized shape to logsumexp.
    dp = (attn * dattn) - (torch.unsqueeze(d, dim=-1) * attn) 
    dq = torch.einsum('brnqk, bnkt -> brnqt', dp, k) 
    dk = torch.einsum('brnkq, brnkt -> bnqt', dp, q)

    ## dsrc, ddest and last part of dk. ## -> Just call pytorch autograd for this.
    affinity.backward(dp.sum(1, keepdim=False))
    dk += k.grad
    dsrc = src.grad
    ddest = dest.grad

    k.grad, src.grad, dest.grad = None, None, None

    return dq.transpose(1, 2), dk, dv, dsrc, ddest


def _debug_triton_fused_gqa_mhsa(q,k,v, backward=False, causal=False):
    ## We clone and detach the tensors for grad checking. ##
    q_torch = q.clone().detach().requires_grad_(True)
    k_torch = k.clone().detach().requires_grad_(True)
    v_torch = v.clone().detach().requires_grad_(True)
    sdpa_output, attn = sdpa_gqa_torch(q_torch, k_torch, v_torch, causal)
    q_sanity = q.clone().detach().requires_grad_(True)
    k_sanity = k.clone().detach().requires_grad_(True)
    v_sanity = v.clone().detach().requires_grad_(True)
    sm_scale = 1.3
    fn = lambda: attention(q, k, v, causal, sm_scale)
    do = torch.randn_like(q).to(q.device)
    triton_output = fn()
    if backward:
        dq_sanity, dk_sanity, dv_sanity = sdpa_gqa_torch_bwd(attn, k_sanity, v_sanity, q_sanity, sdpa_output, do)
        fn = lambda: triton_output.backward(do, retain_graph=True)
        fn()
        sdpa_output.backward(do, retain_graph=True)

    print(f'outputs allclose: {torch.allclose(sdpa_output, triton_output, atol=1e-1, rtol=1e-1)}')
    if backward:
        print(f'dq allclose: {torch.allclose(q_torch.grad, q.grad, atol=1e-1, rtol=1e-1)}')
        print(f'dk allclose: {torch.allclose(k_torch.grad, k.grad, atol=1, rtol=1)}')
        print(f'dv allclose: {torch.allclose(v_torch.grad, v.grad, atol=1, rtol=1)}')

def _debug_triton_universal_attention(q,k,v,static_src,static_dest,backward=False,causal=True):
    assert q.shape[2] % 16 == 0 and k.shape[2] % 16 == 0 and v.shape[2] % 16 == 0 and static_src.shape[2] % 16 == 0 and static_dest.shape[2] % 16 == 0, 'Seq length should be divisible by 16.' 
    q_torch = q.clone().detach().requires_grad_(True)
    k_torch = k.clone().detach().requires_grad_(True)
    v_torch = v.clone().detach().requires_grad_(True)
    static_src_torch = static_src.clone().detach().requires_grad_(True)
    static_dest_torch = static_dest.clone().detach().requires_grad_(True)
    c_, _c = 16, 16
    n_, _n = ceil(q.shape[2] / c_), ceil(q.shape[2] / _c)
    q_torch = torch.reshape(q_torch, (q_torch.shape[0], q_torch.shape[1] // k_torch.shape[1], k_torch.shape[1], _n, _c, q_torch.shape[-1]))
    q_torch = q_torch.transpose(1, 2)
    k_torch = torch.reshape(k_torch, (k_torch.shape[0], k_torch.shape[1], n_, c_, k_torch.shape[-1]))
    v_torch = torch.reshape(v_torch, (v_torch.shape[0], v_torch.shape[1], n_, c_, v_torch.shape[-1]))
    static_src_torch = torch.reshape(static_src_torch, (static_src_torch.shape[0], static_src_torch.shape[1], n_, c_))
    static_dest_torch = torch.reshape(static_dest_torch, (static_dest_torch.shape[0], static_dest_torch.shape[1], _n, _c))
    out, denom = universal_attention_forward(k_torch, v_torch, q_torch,static_src=static_src_torch,static_dest=static_dest_torch)
    torch_output = out.mul(denom.softmax(dim=-1).unsqueeze(-2)).sum(-1)
    sm_scale = 1.3
    fn = lambda: attention(q, k, v, causal, sm_scale, static_src, static_dest)
    triton_output = fn()
    do_torch = torch.randn_like(torch_output).to(q.device)
    print(f'outputs allclose: {torch.allclose(torch.nan_to_num(triton_output), torch.nan_to_num(torch_output.view(triton_output.shape)), atol=1e-1, rtol=1e-1)}')
    if backward:
        q_torch.retain_grad()
        k_torch.retain_grad()
        v_torch.retain_grad()
        static_src_torch.retain_grad()
        static_dest_torch.retain_grad()
        torch_output.backward(do_torch) ## (b,h,r,l,d)
        do = do_torch.clone().detach().requires_grad_(True)
        do = do.transpose(1, 2)
        do = torch.reshape(do, (do.shape[0], do.shape[1] * do.shape[2], do.shape[3], do.shape[4]))
        triton_output.backward(do)
        print('------custom-------')
        print(f'dq allclose: {torch.allclose(torch.nan_to_num(q_torch.grad).reshape(q.grad.shape), torch.nan_to_num(q.grad), atol=1e-1, rtol=1e-1)}')
        print(f'dv allclose: {torch.allclose(torch.nan_to_num(v_torch.grad).reshape(v.grad.shape), torch.nan_to_num(v.grad), atol=1e-1, rtol=1e-1)}')
        print(f'dk allclose: {torch.allclose(torch.nan_to_num(k_torch.grad).reshape(k.grad.shape), torch.nan_to_num(k.grad), atol=1e-1, rtol=1e-1)}')
        print(f'dsrc allclose: {torch.allclose(torch.nan_to_num(static_src_torch.grad).reshape(static_src.grad.shape), torch.nan_to_num(static_src.grad), atol=1e-1, rtol=1e-1)}')
        print(f'ddest allclose: {torch.allclose(torch.nan_to_num(static_dest_torch.grad).reshape(static_dest.grad.shape), torch.nan_to_num(static_dest.grad), atol=1e-1, rtol=1e-1)}')

def _speed_triton_universal_attention(q,k,v,static_src,static_dest,backward=False,causal=True):
    assert q.shape[2] % 16 == 0 and k.shape[2] % 16 == 0 and v.shape[2] % 16 == 0 and static_src.shape[2] % 16 == 0 and static_dest.shape[2] % 16 == 0, 'Seq length should be divisible by 16.' 
    q_torch = q.clone().detach().requires_grad_(True)
    k_torch = k.clone().detach().requires_grad_(True)
    v_torch = v.clone().detach().requires_grad_(True)
    static_src = static_src.clone().detach().requires_grad_(True)
    static_dest = static_dest.clone().detach().requires_grad_(True)
    c_, _c = 32, 32
    n_, _n = ceil(q.shape[2] / c_), ceil(q.shape[2] / _c)
    q_torch = torch.reshape(q_torch, (q_torch.shape[0], q_torch.shape[1] // k_torch.shape[1], k_torch.shape[1], _n, _c, q_torch.shape[-1]))
    q_torch = q_torch.transpose(1, 2)
    k_torch = torch.reshape(k_torch, (k_torch.shape[0], k_torch.shape[1], n_, c_, k_torch.shape[-1]))
    v_torch = torch.reshape(v_torch, (v_torch.shape[0], v_torch.shape[1], n_, c_, v_torch.shape[-1]))
    static_src_torch = torch.reshape(static_src, (static_src.shape[0], static_src.shape[1], n_, c_))
    static_dest_torch = torch.reshape(static_dest, (static_dest.shape[0], static_dest.shape[1], _n, _c))
    for _ in range(5):
        out, denom = universal_attention_forward(k_torch, v_torch, q_torch,static_src=static_src_torch,static_dest=static_dest_torch)
        torch_output = out.mul(denom.softmax(dim=-1).unsqueeze(-2)).sum(-1)

    torch.cuda.synchronize()
    fwd_ua_start = time.time()
    for _ in range(10):
        out, denom = universal_attention_forward(k_torch, v_torch, q_torch,static_src=static_src_torch,static_dest=static_dest_torch)
        torch_output = out.mul(denom.softmax(dim=-1).unsqueeze(-2)).sum(-1)
    torch.cuda.synchronize()
    fwd_ua_end = time.time()

    sm_scale = 1.3
    fn = lambda: attention(q, k, v, causal, sm_scale, static_src, static_dest)
    for _ in range(5):
        triton_output = fn()

    torch.cuda.synchronize()
    fwd_triton_start = time.time()
    for _ in range(10):
        triton_output = fn() 
    torch.cuda.synchronize()
    fwd_triton_end = time.time()

    do_torch = torch.randn_like(torch_output).to(q.device)
    if backward:
        q_torch.retain_grad()
        k_torch.retain_grad()
        v_torch.retain_grad()
        static_src_torch.retain_grad()
        static_dest_torch.retain_grad()
        for _ in range(5):
            torch_output.backward(do_torch, retain_graph=True) ## (b,h,r,l,d)

        torch.cuda.synchronize()
        torch_ua_bwd_start = time.time()
        for _ in range(10):
            torch_output.backward(do_torch, retain_graph=True) ## (b,h,r,l,d)
        torch.cuda.synchronize()
        torch_ua_bwd_end = time.time()

        do = do_torch.clone().detach().requires_grad_(True)
        do = do.transpose(1, 2)
        do = torch.reshape(do, (do.shape[0], do.shape[1] * do.shape[2], do.shape[3], do.shape[4]))

        for _ in range(5):
            triton_output.backward(do, retain_graph=True)

        torch.cuda.synchronize()
        triton_ua_bwd_start = time.time()
        for _ in range(10):
            triton_output.backward(do, retain_graph=True)
        triton_ua_bwd_end = time.time()

        print(f'torch-ua-fwd: {fwd_ua_end-fwd_ua_start}')
        print(f'triton-ua-fwd: {fwd_triton_end-fwd_triton_start}')
        print(f'torch-ua-bwd: {torch_ua_bwd_end-torch_ua_bwd_start}')
        print(f'triton-ua-bwd: {triton_ua_bwd_end-triton_ua_bwd_start}')

def test_case(BATCH, Q_H, KV_H, N_CTX, HEAD_DIM, backward=False):
    print(f'--------test_case BATCH={BATCH} Q_H={Q_H} KV_H={KV_H} N_CTX={N_CTX} HEAD_DIM={HEAD_DIM}---------')
    causal = True
    device="cuda" if torch.cuda.is_available() else "cpu"
    dtype=torch.float16
    q = torch.randn((BATCH, Q_H * KV_H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    k = torch.randn((BATCH, KV_H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    v = torch.randn((BATCH, KV_H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    _debug_triton_fused_gqa_mhsa(q,k,v,backward=backward,causal=causal)

def test_case_universal_attention(BATCH, Q_H, KV_H, N_CTX, HEAD_DIM, backward=False):
    print(f'--------test_case BATCH={BATCH} Q_H={Q_H} KV_H={KV_H} N_CTX={N_CTX} HEAD_DIM={HEAD_DIM}---------')
    causal = True
    device="cuda" if torch.cuda.is_available() else "cpu"
    dtype=torch.float32
    q = torch.rand((BATCH, Q_H * KV_H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    k = torch.rand((BATCH, KV_H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    v = torch.rand((BATCH, KV_H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    static_src = torch.rand((BATCH, KV_H, N_CTX), dtype=dtype, device=device, requires_grad=True)
    static_dest = torch.rand((BATCH, KV_H, N_CTX), dtype=dtype, device=device, requires_grad=True)
    _debug_triton_universal_attention(q,k,v,static_src,static_dest,backward=backward,causal=causal)

def speed_test_ua(BATCH, Q_H, KV_H, N_CTX, HEAD_DIM, backward=True):
    print(f'--------SPEED-TEST BATCH={BATCH} Q_H={Q_H} KV_H={KV_H} N_CTX={N_CTX} HEAD_DIM={HEAD_DIM}---------')
    causal = True
    device="cuda" if torch.cuda.is_available() else "cpu"
    #dtype=torch.bfloat16
    dtype=torch.float32
    q = torch.rand((BATCH, Q_H * KV_H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    k = torch.rand((BATCH, KV_H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    v = torch.rand((BATCH, KV_H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    static_src = torch.rand((BATCH, KV_H, N_CTX), dtype=dtype, device=device, requires_grad=True)
    static_dest = torch.rand((BATCH, KV_H, N_CTX), dtype=dtype, device=device, requires_grad=True)
    _speed_triton_universal_attention(q,k,v,static_src,static_dest,backward=backward,causal=causal)

if __name__ == '__main__':
    ## Here we call whatever we would like to debug. We instantiate with debug sized tensors. ##
    torch.manual_seed(0)
    ## Sample configuration. 8 total query heads and 4 kv heads. ##
    BATCH=1
    Q_H=2 ## Toggle to 1 to have normal MHSA.
    KV_H=4
    N_CTX=16
    HEAD_DIM=16
    ## Test case called with following params:
    ##  1. BATCH
    ##  2. Q_H -> Number of query-head groups (Set to 1 for MHSA).
    ##  3. KV_H -> Number of KV_head groups.
    ##  4. N_CTX -> context length.
    ##  5. HEAD_DIM -> Should be power of two from 32 -> 128 only.
    #test_case_universal_attention(1, 1, 1, 16, 16, backward=False)
    #test_case_universal_attention(1, 1, 1, 32, 16, backward=False)
    #test_case_universal_attention(1, 1, 1, 256, 128, backward=False)
    #test_case_universal_attention(1, 1, 1, 1024, 128, backward=False)
    #test_case_universal_attention(6, 1, 1, 1024, 128, backward=False)
    #test_case_universal_attention(6, 1, 4, 1024, 128, backward=False)
    #test_case_universal_attention(6, 2, 4, 1024, 128, backward=False)
    #test_case_universal_attention(1, 2, 4, 16, 16, backward=False) ## For debugging only, test case that is failing at the moment.
    test_case_universal_attention(1, 1, 1, 16, 16, backward=True)
    #test_case_universal_attention(6, 1, 4, 1024, 128, backward=True) 
    #speed_test_ua(6, 1, 4, 1024, 128, backward=True) 
    #speed_test_ua(1, 1, 32, 2048, 128, backward=True) 
    #test_case_universal_attention(6, 2, 4, 1024, 128, backward=True) 

    ## This tests GQA implementation as we incrementally built from there.. Deprecated now...##
   # test_case(1, 2, 4, 16, 16, backward=False)
   # #test_case(1, 2, 4, 17, 16, backward=False)
   # test_case(32, 2, 4, 16, 16, backward=False)
   # test_case(2, 2, 4, 32, 128, backward=False)
   # test_case(2, 2, 4, 128, 128, backward=False)
   # test_case(1, 2, 4, 1024, 16, backward=False)
   # #test_case(1, 2, 4, 1300, 16, backward=False)
   # test_case(32, 2, 4, 1024, 16, backward=False)
   # test_case(32, 2, 4, 1024, 128, backward=False)
   # test_case(2, 8, 4, 4096, 128, backward=False)
   # test_case(2, 1, 32, 4096, 128, backward=False)
   # test_case(1, 2, 4, 16, 16, backward=True)
   # test_case(1, 2, 4, 32, 16, backward=True)
   # test_case(1, 2, 4, 1024, 128, backward=True)
   # #test_case(1, 2, 4, 1300, 16, backward=True)
   # test_case(32, 2, 4, 16, 16, backward=True)
   # test_case(32, 2, 4, 1024, 16, backward=True)
   # test_case(32, 2, 4, 1024, 128, backward=True)
   # test_case(2, 8, 4, 4096, 128, backward=True)
   # test_case(2, 1, 32, 4096, 128, backward=True)

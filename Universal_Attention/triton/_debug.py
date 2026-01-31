## File that has some debugging helpe functions. ##
import torch
from torch.nn import functional as F
import os
from universal_attention_kernel_opt import attention, _gen_affinity_scores as GenAffTorch
from math import sqrt
from universal_attention_kernel import universal_attention_forward
from math import ceil
import time
from torch.nn.attention.flex_attention import flex_attention
from affinity_kernel import _gen_affinity_scores as AffKern
from affinity_kernel_opt import _gen_affinity_scores as AffKernOpt
from _affinity_generation import _gen_affinity_scores as AffKernOptV2, _affinity_fwd, _affinity_bwd

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
    ## Gen affinity scores already does this, so there's no need for this right now. ##
    #qk += torch.triu(torch.full(qk.shape, -1e6), diagonal=1).to(qk.device)

    affinity = GenAffTorch(k, src, dest)
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
    dkt, dsrc, ddest = torch.autograd.grad(affinity, [k, src, dest], grad_outputs=dp.sum(1, keepdim=False))
    dk += dkt

    return torch.reshape(dq, incoming_query_shape), dk, dv, dsrc, ddest

def make_universal_score_mod(k: torch.Tensor, src: torch.Tensor, dest: torch.Tensor):
    """Return a score_mod callable for flex_attention that reproduces the
    universal-attention affinity computed by _gen_affinity_scores.

    The returned callable closes over a precomputed affinity tensor to avoid
    recomputing O(N^2) work per callback.
    """
    affs = GenAffTorch(k, src, dest)
    scale_fix = sqrt(k.shape[-1]) 

    def score_mod(score, b: int, h: int, q_idx: int, k_idx: int):
        return score*scale_fix + affs[b, h, q_idx, k_idx]

    return score_mod


def ua_flex_attention(q, k, v, src, dest):
    if k.shape[1] != q.shape[1]:
        ## Then this is GQA case, we do something naive to test correctness. ##
        k = k.repeat(1, q.shape[1]//k.shape[1], 1, 1)
        v = v.repeat(1, q.shape[1]//v.shape[1], 1, 1)
        src = src.repeat(1, q.shape[1]//src.shape[1], 1)
        dest = dest.repeat(1, q.shape[1]//dest.shape[1], 1)
    score_mod = make_universal_score_mod(k, src, dest)
    # _gen_affinity_scores already encodes causal structure; keep is_causal=False.
    return flex_attention(q, k, v, score_mod=score_mod)

def ua_sdpa(q, k, v, src, dest):
    if k.shape[1] != q.shape[1]:
        ## This is the GQA case. ##
        r = q.shape[1] // k.shape[1]

    affs = GenAffTorch(k, src, dest, r)
    torch.backends.cuda.enable_math_sdp(False)
    attn = F.scaled_dot_product_attention(
        q, 
        k.repeat(1, r, 1, 1),
        v.repeat(1, r, 1, 1),
        attn_mask=affs,
        scale=1,
    )  # b h l d
    return attn

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


def _profile_triton_affinity_creation(q,k,v,static_src,static_dest,backward=False,causal=True):
    q_torch = q.clone().detach().requires_grad_(True)
    k_torch = k.clone().detach().requires_grad_(True)
    v_torch = v.clone().detach().requires_grad_(True)
    static_src_torch = static_src.clone().detach().requires_grad_(True)
    static_dest_torch = static_dest.clone().detach().requires_grad_(True)
    affinities_torch = _gen_affinity_scores(k_torch,src=static_src_torch,dest=static_dest_torch)
    sm_scale = 1.3
    k_v1 = k.clone().detach().requires_grad_(True)
    static_src_v1 = static_src.clone().detach().requires_grad_(True)
    static_dest_v1 = static_dest.clone().detach().requires_grad_(True)
    fn_opt_v2 = lambda: AffKernOptV2(k, static_src, static_dest)
    fn_opt = lambda: AffKernOpt(k_v1, static_src_v1, static_dest_v1)
    triton_out = fn_opt_v2()
    triton_out_v1 = fn_opt()
    do_torch = torch.randn_like(affinities_torch).to(q.device)
    do = do_torch.clone().detach()
    print(f'fwd pass: {torch.allclose(triton_out, affinities_torch, atol=1e-1, rtol=1e-1)}')
    print(f'fwd pass (triton v. triton): {torch.allclose(triton_out, triton_out_v1, atol=1e-1, rtol=1e-1)}')
    if backward:
        affinities_torch.backward(do_torch)
        triton_out.backward(do)
        triton_out_v1.backward(do)

        ## Some correctness checks. ##
        print(f'bwd pass - k: {torch.allclose(k_torch.grad, k.grad, atol=1e-1, rtol=1e-1)}')
        print(f'bwd pass - src: {torch.allclose(static_src_torch.grad, static_src.grad, atol=1e-1, rtol=1e-1)}')
        print(f'bwd pass - dest: {torch.allclose(static_dest_torch.grad, static_dest.grad, atol=1e-1, rtol=1e-1)}')

        print(f'bwd pass - k (triton v. triton): {torch.allclose(k_v1.grad, k.grad, atol=1e-1, rtol=1e-1)}')
        print(f'bwd pass - src (triton v. triton): {torch.allclose(static_src_v1.grad, static_src.grad, atol=1e-1, rtol=1e-1)}')
        print(f'bwd pass - dest (triton v. triton): {torch.allclose(static_dest_v1.grad, static_dest.grad, atol=1e-1, rtol=1e-1)}')

def _speed_test_triton_affinity_creation(q,k,v,static_src,static_dest,backward=False,causal=True):
    q_torch = q.clone().detach().requires_grad_(True)
    k_torch = k.clone().detach().requires_grad_(True)
    v_torch = v.clone().detach().requires_grad_(True)
    static_src_torch = static_src.clone().detach().requires_grad_(True)
    static_dest_torch = static_dest.clone().detach().requires_grad_(True)
    affinities_torch = _gen_affinity_scores(k_torch,src=static_src_torch,dest=static_dest_torch)
    do = torch.randn_like(affinities_torch).to(q.device)

    ## Quick dirty speed tests. ##
    for _ in range(5):
        affinities_torch = _gen_affinity_scores(k_torch,src=static_src_torch,dest=static_dest_torch)
        affinities_torch.backward(do)

    torch.cuda.synchronize()
    torch_start_fwd = time.time()
    for _ in range(10):
        affinities_torch = _gen_affinity_scores(k_torch,src=static_src_torch,dest=static_dest_torch)
    torch.cuda.synchronize()
    torch_end_fwd = time.time()

    torch.cuda.synchronize()
    torch_start_bwd = time.time()
    for _ in range(10):
        affinities_torch.backward(do, retain_graph=True)
    torch.cuda.synchronize()
    torch_end_bwd = time.time()

    fn_opt = lambda: AffKernOpt(k, static_src, static_dest)
    fn_opt_v2 = lambda: AffKernOptV2(k, static_src, static_dest)

    for _ in range(5):
        v1_out = fn_opt()
        v1_out.backward(do)

    torch.cuda.synchronize()
    triton_v1_start_fwd = time.time()
    for _ in range(10):
        v1_out = fn_opt()
    torch.cuda.synchronize()
    triton_v1_end_fwd = time.time()

    torch.cuda.synchronize()
    triton_v1_start_bwd = time.time()
    for _ in range(10):
        v1_out.backward(do, retain_graph=True)
    torch.cuda.synchronize()
    triton_v1_end_bwd = time.time()

    for _ in range(5):
        v2_out = fn_opt_v2()
        v2_out.backward(do)

    torch.cuda.synchronize()
    triton_v2_start_fwd = time.time()
    for _ in range(10):
        v2_out = fn_opt_v2()
    torch.cuda.synchronize()
    triton_v2_end_fwd = time.time()

    torch.cuda.synchronize()
    triton_v2_start_bwd = time.time()
    for _ in range(10):
        v2_out.backward(do, retain_graph=True)
    triton_v2_end_bwd = time.time()
    torch.cuda.synchronize()

    print(f'Speed test results:')
    print(f'torch fwd time: {torch_end_fwd-torch_start_fwd}')
    print(f'torch bwd time: {torch_end_bwd-torch_start_bwd}')
    print(f'triton v1 fwd time: {triton_v1_end_fwd-triton_v1_start_fwd}')
    print(f'triton v1 bwd time: {triton_v1_end_bwd-triton_v1_start_bwd}')
    print(f'triton v2 fwd time: {triton_v2_end_fwd-triton_v2_start_fwd}')
    print(f'triton v2 bwd time: {triton_v2_end_bwd-triton_v2_start_bwd}')
    print(f'torch e2e time: {(torch_end_fwd-torch_start_fwd)+(torch_end_bwd-torch_start_bwd)}')
    print(f'triton v1 e2e time: {(triton_v1_end_fwd-triton_v1_start_fwd)+(triton_v1_end_bwd-triton_v1_start_bwd)}')
    print(f'triton v2 e2d time: {(triton_v2_end_fwd-triton_v2_start_fwd)+(triton_v2_end_bwd-triton_v2_start_bwd)}')

def helper_which_notclose(one: torch.tensor, two: torch.tensor, atol: float, rtol: float):
    diffs = torch.abs(one - two)

    idx = diffs > rtol * torch.abs(two) + atol

    print(f'one: {one[idx]}\n, two: {two[idx]}')
    print(f'idxs: {idx.nonzero()}\n idxs size: {idx.nonzero().shape}')

    return idx

def _debug_affinity_generation(k,static_src,static_dest):
    k_torch = k.clone().detach().requires_grad_(True)
    static_src_torch = static_src.clone().detach().requires_grad_(True)
    static_dest_torch = static_dest.clone().detach().requires_grad_(True)

    torch_output = GenAffTorch(k_torch, static_src_torch, static_dest_torch)
    custom_output = _affinity_fwd(k, static_src, static_dest)
    print(f'outputs allclose-sdpa vs. triton (affinity-only): {torch.allclose(torch.nan_to_num(custom_output), torch.nan_to_num(torch_output), atol=1e-2, rtol=1e-2)}')

    ## make daff. ##
    daff_torch = torch.randn_like(torch_output).to(k.device)
    daff_custom = daff_torch.clone().detach()

    ## Then invoke backwards pass. ##
    torch_output.backward(daff_torch)
    dk, dsrc, ddest = _affinity_bwd(k, static_src, static_dest, daff_custom)
    print(f'dk allclose-sdpa vs. triton (affinity-only): {torch.allclose(torch.nan_to_num(dk), torch.nan_to_num(k_torch.grad), atol=1e-2, rtol=1e-2)}')
    print(f'dsrc allclose-sdpa vs. triton (affinity-only): {torch.allclose(torch.nan_to_num(dsrc), torch.nan_to_num(static_src_torch.grad), atol=1e-2, rtol=1e-2)}')
    print(f'ddest allclose-sdpa vs. triton (affinity-only): {torch.allclose(torch.nan_to_num(ddest), torch.nan_to_num(static_dest_torch.grad), atol=1e-2, rtol=1e-2)}')


def _debug_triton_universal_attention(q,k,v,static_src,static_dest,backward=False,causal=True):
    assert q.shape[2] % 16 == 0 and k.shape[2] % 16 == 0 and v.shape[2] % 16 == 0 and static_src.shape[2] % 16 == 0 and static_dest.shape[2] % 16 == 0, 'Seq length should be divisible by 16.' 
    q_torch = q.clone().detach().requires_grad_(True)
    k_torch = k.clone().detach().requires_grad_(True)
    v_torch = v.clone().detach().requires_grad_(True)
    static_src_torch = static_src.clone().detach().requires_grad_(True)
    static_dest_torch = static_dest.clone().detach().requires_grad_(True)
    q_flex = q.clone().detach().requires_grad_(True)
    k_flex = k.clone().detach().requires_grad_(True)
    v_flex = v.clone().detach().requires_grad_(True)
    static_src_flex = static_src.clone().detach().requires_grad_(True)
    static_dest_flex = static_dest.clone().detach().requires_grad_(True)
    q_sdpa = q.clone().detach().requires_grad_(True)
    k_sdpa = k.clone().detach().requires_grad_(True)
    v_sdpa = v.clone().detach().requires_grad_(True)
    static_src_sdpa = static_src.clone().detach().requires_grad_(True)
    static_dest_sdpa = static_dest.clone().detach().requires_grad_(True)
    c_, _c = 16, 16
    n_, _n = ceil(q.shape[2] / c_), ceil(q.shape[2] / _c)
    q_torch = torch.reshape(q_torch, (q_torch.shape[0], q_torch.shape[1] // k_torch.shape[1], k_torch.shape[1], _n, _c, q_torch.shape[-1]))
    q_torch = q_torch.transpose(1, 2)
    k_torch = torch.reshape(k_torch, (k_torch.shape[0], k_torch.shape[1], n_, c_, k_torch.shape[-1]))
    v_torch = torch.reshape(v_torch, (v_torch.shape[0], v_torch.shape[1], n_, c_, v_torch.shape[-1]))
    static_src_torch = torch.reshape(static_src_torch, (static_src_torch.shape[0], static_src_torch.shape[1], n_, c_))
    static_dest_torch = torch.reshape(static_dest_torch, (static_dest_torch.shape[0], static_dest_torch.shape[1], _n, _c))
    #out, denom = universal_attention_forward(k_torch, v_torch, q_torch,static_src=static_src_torch,static_dest=static_dest_torch)
    #torch_output = out.mul(denom.softmax(dim=-1).unsqueeze(-2)).sum(-1)
    sm_scale = 1.3
    ## We print out stuff and run in interpreter mode to help us debug. ##
    fn = lambda: attention(q, k, v, causal, sm_scale, static_src, static_dest)
    triton_output = fn()
    ## Temporarily comment out flex attention to remove the need to make it support GQA. ##
    sdpa_attention_output = ua_sdpa(q_sdpa, k_sdpa, v_sdpa, static_src_sdpa, static_dest_sdpa)
    do_torch = torch.randn_like(triton_output).to(q.device)
    print(f'outputs allclose-sdpa vs. triton: {torch.allclose(torch.nan_to_num(triton_output), torch.nan_to_num(sdpa_attention_output), atol=1e-2, rtol=1e-2)}')
    ## For debugging purposes only. ##
    if backward:
        q_torch.retain_grad()
        k_torch.retain_grad()
        v_torch.retain_grad()
        static_src_torch.retain_grad()
        static_dest_torch.retain_grad()
        do = do_torch.clone().detach().requires_grad_(True)
        triton_output.backward(do)
        do_sdpa = do.clone().detach().requires_grad_(True)
        sdpa_attention_output.backward(do_sdpa)
        print('-----sdpa_attn-------')
        print(f'dq allclose-flex: {torch.allclose(torch.nan_to_num(q.grad), torch.nan_to_num(q_sdpa.grad), atol=1e-2, rtol=1e-2)}')
        print(f'dv allclose-flex: {torch.allclose(torch.nan_to_num(v.grad), torch.nan_to_num(v_sdpa.grad), atol=1e-2, rtol=1e-2)}')
        print(f'dk allclose-flex: {torch.allclose(torch.nan_to_num(k.grad), torch.nan_to_num(k_sdpa.grad), atol=1e-2, rtol=1e-2)}')
        print(f'dsrc allclose-flex: {torch.allclose(torch.nan_to_num(static_src.grad), torch.nan_to_num(static_src_sdpa.grad), atol=1e-2, rtol=1e-2)}')
        print(f'ddest allclose-flex: {torch.allclose(torch.nan_to_num(static_dest.grad), torch.nan_to_num(static_dest_sdpa.grad), atol=1e-2, rtol=1e-2)}')

def _speed_triton_universal_attention(q,k,v,static_src,static_dest,backward=False,causal=True):
    assert q.shape[2] % 16 == 0 and k.shape[2] % 16 == 0 and v.shape[2] % 16 == 0 and static_src.shape[2] % 16 == 0 and static_dest.shape[2] % 16 == 0, 'Seq length should be divisible by 16.' 
    q_torch = q.clone().detach().requires_grad_(True)
    k_torch = k.clone().detach().requires_grad_(True)
    v_torch = v.clone().detach().requires_grad_(True)
    static_src = static_src.clone().detach().requires_grad_(True)
    static_dest = static_dest.clone().detach().requires_grad_(True)
    q_sdpa = q.clone().detach().requires_grad_(True)
    k_sdpa = k.clone().detach().requires_grad_(True)
    v_sdpa = v.clone().detach().requires_grad_(True)
    static_src_sdpa = static_src.clone().detach().requires_grad_(True)
    static_dest_sdpa = static_dest.clone().detach().requires_grad_(True)
    c_, _c = 32, 32
    n_, _n = ceil(q.shape[2] / c_), ceil(q.shape[2] / _c)
    q_torch = torch.reshape(q_torch, (q_torch.shape[0], q_torch.shape[1] // k_torch.shape[1], k_torch.shape[1], _n, _c, q_torch.shape[-1]))
    q_torch = q_torch.transpose(1, 2)
    k_torch = torch.reshape(k_torch, (k_torch.shape[0], k_torch.shape[1], n_, c_, k_torch.shape[-1]))
    v_torch = torch.reshape(v_torch, (v_torch.shape[0], v_torch.shape[1], n_, c_, v_torch.shape[-1]))
    static_src_torch = torch.reshape(static_src, (static_src.shape[0], static_src.shape[1], n_, c_))
    static_dest_torch = torch.reshape(static_dest, (static_dest.shape[0], static_dest.shape[1], _n, _c))

    sdpa_out = ua_sdpa(q_sdpa, k_sdpa, v_sdpa, static_src_sdpa, static_dest_sdpa)

    do = torch.randn_like(sdpa_out).to(q.device)
    do_sdpa = do.clone().detach().requires_grad_(True)

    start_triton = torch.cuda.Event(enable_timing=True)
    end_triton = torch.cuda.Event(enable_timing=True)
    start_sdpa = torch.cuda.Event(enable_timing=True)
    end_sdpa = torch.cuda.Event(enable_timing=True)
    for _ in range(5):
        sdpa_out = ua_sdpa(q_sdpa, k_sdpa, v_sdpa, static_src_sdpa, static_dest_sdpa)
        sdpa_out.backward(do_sdpa)

    torch.cuda.synchronize()
    start_sdpa.record()
    for _ in range(10):
        sdpa_out = ua_sdpa(q_sdpa, k_sdpa, v_sdpa, static_src_sdpa, static_dest_sdpa)
        sdpa_out.backward(do_sdpa)
    end_sdpa.record()

    torch.cuda.synchronize()


    sm_scale = 1.3
    fn = lambda: attention(q, k, v, causal, sm_scale, static_src, static_dest, True, True)
    for _ in range(5):
        triton_output = fn()
        triton_output.backward(do)

    torch.cuda.synchronize()
    start_triton.record()
    for _ in range(10):
        triton_output = fn() 
        triton_output.backward(do)
    end_triton.record()
    torch.cuda.synchronize()

    print(f'triton-ua: {start_triton.elapsed_time(end_triton)}')
    print(f'sdpa-ua: {start_sdpa.elapsed_time(end_sdpa)}')

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
    #dtype=torch.float32
    is_interpret = os.environ.get("TRITON_INTERPRET") == "1"
    if is_interpret:
        dtype=torch.float32
    else:
        dtype=torch.bfloat16
    q = torch.rand((BATCH, Q_H * KV_H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    k = torch.rand((BATCH, KV_H, N_CTX, HEAD_DIM), dtype=dtype, device=device) 
    k /= k.pow(2).sum(-1, True).sqrt().add(1e-6)
    k.requires_grad_(True)
    v = torch.rand((BATCH, KV_H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    static_src = torch.rand((BATCH, KV_H, N_CTX), dtype=dtype, device=device).sigmoid()
    static_src.requires_grad_(True)
    static_dest = torch.rand((BATCH, KV_H, N_CTX), dtype=dtype, device=device).sigmoid()
    static_dest.requires_grad_(True)
    _debug_triton_universal_attention(q,k,v,static_src,static_dest,backward=backward,causal=causal)
    #k_affinity = k.clone().detach().requires_grad_(True)
    #src_affinity = static_src.clone().detach().requires_grad_(True)
    #dest_affinity = static_dest.clone().detach().requires_grad_(True)
    #_debug_affinity_generation(k_affinity, src_affinity, dest_affinity)

def profile_affinity_creation(BATCH, Q_H, KV_H, N_CTX, HEAD_DIM, backward=False):
    ## This is a profiling pass, to ensure that we launch singular kernels only.
    print(f'--------test_case BATCH={BATCH} Q_H={Q_H} KV_H={KV_H} N_CTX={N_CTX} HEAD_DIM={HEAD_DIM}---------')
    causal = True
    device="cuda" if torch.cuda.is_available() else "cpu"
    #dtype=torch.float32
    dtype=torch.bfloat16
    q = torch.rand((BATCH, Q_H * KV_H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    k = torch.rand((BATCH, KV_H, N_CTX, HEAD_DIM), dtype=dtype, device=device) 
    k /= k.pow(2).sum(-1, True).sqrt().add(1e-6)
    k.requires_grad_(True)
    v = torch.rand((BATCH, KV_H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    static_src = torch.rand((BATCH, KV_H, N_CTX), dtype=dtype, device=device).sigmoid()
    static_src.requires_grad_(True)
    static_dest = torch.rand((BATCH, KV_H, N_CTX), dtype=dtype, device=device).sigmoid()
    static_dest.requires_grad_(True)
    _profile_triton_affinity_creation(q,k,v,static_src,static_dest,backward=backward,causal=causal)


def speed_test_ua(BATCH, Q_H, KV_H, N_CTX, HEAD_DIM, backward=True):
    print(f'--------SPEED-TEST BATCH={BATCH} Q_H={Q_H} KV_H={KV_H} N_CTX={N_CTX} HEAD_DIM={HEAD_DIM}---------')
    causal = True
    device="cuda" if torch.cuda.is_available() else "cpu"
    dtype=torch.bfloat16
    q = torch.rand((BATCH, Q_H * KV_H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    k = torch.rand((BATCH, KV_H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    v = torch.rand((BATCH, KV_H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    static_src = torch.rand((BATCH, KV_H, N_CTX), dtype=dtype, device=device, requires_grad=True)
    static_dest = torch.rand((BATCH, KV_H, N_CTX), dtype=dtype, device=device, requires_grad=True)
    _speed_triton_universal_attention(q,k,v,static_src,static_dest,backward=backward,causal=causal)

def speed_test_affinity_creation(BATCH, Q_H, KV_H, N_CTX, HEAD_DIM, backward=True):
    print(f'--------SPEED-TEST BATCH={BATCH} Q_H={Q_H} KV_H={KV_H} N_CTX={N_CTX} HEAD_DIM={HEAD_DIM}---------')
    causal = True
    device="cuda" if torch.cuda.is_available() else "cpu"
    dtype=torch.bfloat16
    q = torch.rand((BATCH, Q_H * KV_H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    k = torch.rand((BATCH, KV_H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    v = torch.rand((BATCH, KV_H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    static_src = torch.rand((BATCH, KV_H, N_CTX), dtype=dtype, device=device, requires_grad=True)
    static_dest = torch.rand((BATCH, KV_H, N_CTX), dtype=dtype, device=device, requires_grad=True)
    _speed_test_triton_affinity_creation(q,k,v,static_src,static_dest,backward=backward,causal=causal)

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
    ## All TCs passing. ##
    #test_case_universal_attention(1, 1, 1, 128, 128, backward=True)
    #test_case_universal_attention(1, 1, 1, 256, 128, backward=True)
    #test_case_universal_attention(1, 1, 1, 384, 128, backward=True)
    #test_case_universal_attention(1, 1, 1, 512, 128, backward=True)
    #test_case_universal_attention(1, 1, 1, 1024, 128, backward=True)
    #test_case_universal_attention(1, 1, 1, 2048, 128, backward=True)
    #test_case_universal_attention(2, 1, 1, 2048, 128, backward=True)
    #test_case_universal_attention(2, 1, 2, 2048, 128, backward=True)
    #test_case_universal_attention(2, 1, 8, 2048, 128, backward=True)
    #test_case_universal_attention(2, 2, 8, 2048, 128, backward=True)
    #test_case_universal_attention(1, 1, 32, 2048, 128, backward=True)
    #test_case_universal_attention(2, 1, 32, 2048, 128, backward=True) 
    #test_case_universal_attention(8, 4, 4, 4096, 64, backward=True)  ## This is the input we have to figure out where perf has vanished on.
    
    ## Longer sequence tests, need to reduce number of heads for this. ##
    #test_case_universal_attention(1, 1, 16, 4096, 128, backward=True) 
    #test_case_universal_attention(2, 1, 8, 4096, 128, backward=True) 

    ## Now, some grouped query attention tests as well. ##
    #test_case_universal_attention(1, 2, 2, 128, 128, backward=True)
    #test_case_universal_attention(1, 1, 4, 128, 128, backward=True)
    #test_case_universal_attention(1, 2, 8, 512, 128, backward=True)
    #test_case_universal_attention(1, 2, 16, 2048, 128, backward=True)
    #test_case_universal_attention(2, 2, 16, 2048, 128, backward=True) 
    #test_case_universal_attention(2, 2, 16, 1024, 128, backward=True) 
    #test_case_universal_attention(8, 4, 4, 4096, 128, backward=True) 
    ## More custom configs to enhance Davis' experience in using this kernel. ##

    #test_case_universal_attention(4, 4, 5, 4096, 64, backward=True) 
    #test_case_universal_attention(4, 4, 5, 4096, 80, backward=True) 
    #test_case_universal_attention(4, 4, 4, 4096, 64, backward=True) 
    #test_case_universal_attention(4, 4, 4, 4096, 80, backward=True) 

    #test_case_universal_attention(4, 4, 5, 1024, 64, backward=True) 
    #test_case_universal_attention(4, 4, 4, 1024, 64, backward=True) 
    #test_case_universal_attention(4, 4, 4, 1024, 80, backward=True) 
    
    ## SPEED TESTS TO ASSESS PERFORMANCE ##
    #speed_test_ua(2, 1, 32, 256, 128, backward=True) 
    #speed_test_ua(2, 1, 32, 512, 128, backward=True) 
    #speed_test_ua(2, 1, 32, 1024, 128, backward=True) 
    #speed_test_ua(2, 4, 8, 1024, 128, backward=True)  ## -> This is Llama 3.1-8bn's configuration.
    #speed_test_ua(4, 4, 8, 1024, 128, backward=True)  ## -> This is Llama 3.1-8bn's configuration that I am testing end-to-end.
    #speed_test_ua(4, 4, 4, 2048, 64, backward=True)  
    #speed_test_ua(4, 4, 4, 2048, 80, backward=True)  
    #speed_test_ua(4, 4, 5, 2048, 80, backward=True)  
    #speed_test_ua(1, 4, 5, 4096, 64, backward=True)  

    speed_test_ua(8, 4, 4, 4096, 64, backward=True)  # The important config to test on.
    #speed_test_ua(8, 4, 4, 4096, 80, backward=True)  # The important config to test on.
    #speed_test_ua(8, 4, 4, 4096, 128, backward=True)  # The important config to test on.

    #speed_test_ua(2, 1, 32, 1024, 128, backward=True) 

    #profile_affinity_creation(1, 1, 8, 4096, 128, backward=True)
    #profile_affinity_creation(1, 1, 8, 128, 128, backward=True)
    #speed_test_affinity_creation(8, 1, 8, 2048, 128, backward=True)

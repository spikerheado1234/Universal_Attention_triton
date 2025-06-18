import torch
from universal_attention_autograd import UniversalAttention as uni_attn_torch
from universal_attention_autograd import UniversalAttention2 as uni_attn_torch_2
from universal_attention_autograd import SMVecMatMul

def main(config):
    b       = config['batch_size']
    s       = config['seq_len']
    n       = config['n_heads']
    n_kv    = config['n_kv_heads']
    rep     = config['n_rep']
    d       = config['head_dim']
    c       = config['chunk_size']
    n_c     = config['n_chunks']
    device  = config['device']

    query_state = torch.rand(b, s, n, d, device=device)
    key_state   = torch.rand(b, s, n_kv, d, device=device)
    value_state = torch.rand(b, s, n_kv, d, device=device)
    static      = torch.rand(b, s, 2 * n_kv, device=device)

    # Reshape for the kernels
    xq = query_state.transpose(1, 2).view(b, n_kv, rep, s, d)
    kc = key_state.transpose(1, 2).view(b, n_kv, n_c, c, d)
    vc = value_state.transpose(1, 2).view(b, n_kv, n_c, c, d)
    # kt = key_state.transpose(1, 2).transpose(-2,-1)
    static = static.view(b, s, 2, n_kv).permute(2, 0, 3, 1)
    static_src = static[0].view(b, n_kv, n_c, c)
    static_dest = static[1] 

    UA_impl = {
        "pytorch_base": uni_attn_torch.apply,
        "pytorch_chunked": uni_attn_torch_2.apply,
        # "triton": uni_attn_triton.apply,
    }
    SMVMM = SMVecMatMul.apply

    results = {}

    for key in UA_impl.keys():
        kc_          = kc.clone().detach().to(device).requires_grad_()
        vc_          = vc.clone().detach().to(device).requires_grad_()
        static_src_  = static_src.clone().detach().to(device).requires_grad_()

        if key == "pytorch_chunked":
            xq_ = xq.view(b, n_kv, rep, n_c, c, d).clone().detach().to(device).requires_grad_()
            static_dest_ = static_dest.view(b, n_kv, n_c, c).clone().detach().to(device).requires_grad_()
        else:
            xq_ = xq.clone().detach().to(device).requires_grad_()
            static_dest_ = static_dest.clone().detach().to(device).requires_grad_()
        
        # Forward
        output, denom = UA_impl[key](kc_, vc_, xq_, static_src_, static_dest_)
        out = SMVMM(output, denom)

        # Backward
        output.retain_grad()
        denom.retain_grad()
        loss = out.pow(2).sum() # some random loss to enable autograd
        loss.backward()

        if key == "pytorch_chunked":
            xq_grad = xq_.grad.view(b, n_kv, rep, s, d).clone().detach().cpu()
            static_dest_grad = static_dest_.grad.view(b, n_kv, s).clone().detach().cpu()
        else:
            xq_grad = xq_.grad.clone().detach().cpu()
            static_dest_grad = static_dest_.grad.clone().detach().cpu()

        results[key] = {
            'output': output.clone().detach().cpu(),
            'denom': denom.clone().detach().cpu(),
            'kc.grad': kc_.grad.clone().detach().cpu(),
            'vc.grad': vc_.grad.clone().detach().cpu(),
            'xq.grad': xq_grad,
            'static_src.grad': static_src_.grad.clone().detach().cpu(),
            'static_dest.grad': static_dest_grad,
        }

        del output, denom, kc_, vc_, xq_, static_src_, static_dest_, xq_grad, static_dest_grad

    # Compare results
    print("Check results:")
    for attr in ['output', 'denom', 'kc.grad', 'vc.grad', 'xq.grad',
            'static_src.grad', 'static_dest.grad']:
        r = []
        for key in results.keys():
            r.append(results[key][attr])
        print(f"{attr} max error:", (r[0].float() - r[1].float()).abs().max().item())

if __name__ == "__main__":
    test_config = {
        'batch_size': 16,
        'seq_len': 512,
        'chunk_size': 128,
        'n_heads': 32,
        'n_kv_heads': 8,
        'dim': 256,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    }

    # test_config = {
    #     'batch_size': 2,
    #     'seq_len': 16,
    #     'chunk_size': 4,
    #     'n_heads': 2,
    #     'n_kv_heads': 1,
    #     'dim': 8,
    #     'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    # }

    test_config['n_rep'] = test_config['n_heads'] // test_config['n_kv_heads']
    test_config['head_dim'] = test_config['dim'] // test_config['n_heads']
    test_config['n_chunks'] = test_config['seq_len'] // test_config['chunk_size']

    main(test_config)

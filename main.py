import torch
from universal_attention_autograd import UniversalAttention

def main(config):
    # Generate random input
    b       = config['batch_size']
    s       = config['seq_len']
    n       = config['n_heads']
    n_kv    = config['n_kv_heads']
    rep     = config['n_rep']
    d       = config['head_dim']
    c       = config['chunk_size']
    n_c     = config['n_chunks']
    device  = config['device']

    query_state = torch.rand(b, s, n, d).to(device)
    key_state   = torch.rand(b, s, n_kv, d).to(device)
    value_state = torch.rand(b, s, n_kv, d).to(device)
    static      = torch.rand(b, s, 2 * n_kv).to(device)

    # Reshape for the kernels
    xq = query_state.transpose(1, 2).view(b, n_kv, rep, s, d)
    kc = key_state.transpose(1, 2).view(b, n_kv, n_c, c, d)
    vc = value_state.transpose(1, 2).view(b, n_kv, n_c, c, d)
    # kt = key_state.transpose(1, 2).transpose(-2,-1)
    static = static.view(b, s, 2, n_kv).permute(2, 0, 3, 1)
    static_src = static[0].view(b, n_kv, n_c, c)
    static_dest = static[1] 

    # Perform universal attention
    UA = UniversalAttention.apply

    # Forward
    output, denom = UA(kc, vc, xq, static_src, static_dest)

    print(output.shape, denom.shape)

    # Backward
    loss = torch.sum(output**2) + torch.sum(denom**2) # some random loss to enable autograd
    loss.backward()
    print(output.grad.shape, denom.grad.shape)


if __name__ == "__main__":
    test_config = {
        'batch_size': 16,
        'seq_len': 2048,
        'chunk_size': 256,
        'n_heads': 32,
        'n_kv_heads': 8,
        'dim': 2048,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    }

    test_config['n_rep'] = test_config['n_heads'] // test_config['n_kv_heads']
    test_config['head_dim'] = test_config['dim'] // test_config['n_heads']
    test_config['n_chunks'] = test_config['seq_len'] // test_config['chunk_size']

    main(test_config)

import torch
import sys
sys.path.append('.')

from universal_attention_kernel_opt import attention, _gen_affinity_scores
from _debug import ua_bwd

def minimal_debug_test():
    """Minimal test to isolate the dk gradient issue."""
    torch.manual_seed(42)
    
    # Test with small, controlled inputs
    BATCH, Q_H, KV_H, HEAD_DIM = 1, 1, 1, 64
    device = "cuda"
    dtype = torch.bfloat16
    
    for N_CTX in [128, 256]:
        print(f"\n=== Testing N_CTX={N_CTX} ===")
        
        # Create simple test tensors
        q = torch.randn((BATCH, Q_H * KV_H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        k = torch.randn((BATCH, KV_H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        v = torch.randn((BATCH, KV_H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        static_src = torch.randn((BATCH, KV_H, N_CTX), dtype=dtype, device=device, requires_grad=True)
        static_dest = torch.randn((BATCH, KV_H, N_CTX), dtype=dtype, device=device, requires_grad=True)
        
        # Test Triton implementation
        q_triton = q.clone().detach().requires_grad_(True)
        k_triton = k.clone().detach().requires_grad_(True)
        v_triton = v.clone().detach().requires_grad_(True)
        src_triton = static_src.clone().detach().requires_grad_(True)
        dest_triton = static_dest.clone().detach().requires_grad_(True)
        
        sm_scale = 1.0  # Use simple scale for debugging
        triton_output = attention(q_triton, k_triton, v_triton, True, sm_scale, src_triton, dest_triton)
        
        # Use simple gradient
        do = torch.ones_like(triton_output)
        triton_output.backward(do)
        
        # Test reference implementation  
        q_ref = q.clone().detach().requires_grad_(True)
        k_ref = k.clone().detach().requires_grad_(True) 
        v_ref = v.clone().detach().requires_grad_(True)
        src_ref = static_src.clone().detach().requires_grad_(True)
        dest_ref = static_dest.clone().detach().requires_grad_(True)
        
        dq_ref, dk_ref, dv_ref, dsrc_ref, ddest_ref = ua_bwd(q_ref, k_ref, v_ref, src_ref, dest_ref, do)
        
        # Compare dk gradients
        dk_diff = (k_triton.grad - dk_ref).abs()
        print(f"dk max absolute difference: {dk_diff.max().item():.6f}")
        print(f"dk mean absolute difference: {dk_diff.mean().item():.6f}")
        print(f"dk relative difference: {(dk_diff / dk_ref.abs().clamp(min=1e-6)).max().item():.6f}")
        
        # Check if the difference is significant
        is_close = torch.allclose(k_triton.grad, dk_ref, atol=1e-2, rtol=1e-2)
        print(f"dk gradients close (atol=1e-2, rtol=1e-2): {is_close}")
        
        if not is_close:
            print("FAILED: dk gradients do not match!")
            # Print some sample values for debugging
            print(f"Triton dk[0,0,0:5,0]: {k_triton.grad[0,0,0:5,0]}")
            print(f"Reference dk[0,0,0:5,0]: {dk_ref[0,0,0:5,0]}")
            print(f"Ratio (Triton/Ref): {(k_triton.grad[0,0,0:5,0] / dk_ref[0,0,0:5,0])}")
        else:
            print("PASSED: dk gradients match!")

if __name__ == '__main__':
    minimal_debug_test()

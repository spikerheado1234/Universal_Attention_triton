import triton
import triton.language as tl
import torch
import time
import math


if __name__ == "__main__":
    b, n_kv, m, n = 2, 4, 16384, 16384
    runs = 1000
    dtype = torch.float32

    A = torch.randn(b, n_kv, m, n, device='cuda', dtype=dtype)

    _ = efficient_cumsum(A)
    torch.cuda.synchronize()

    triton_time, torch_time = 0, 0
    for _ in range(runs):
        A = torch.randn(b, n_kv, m, n, device='cuda', dtype=dtype)

        start_time = time.time()
        _ = efficient_cumsum(A)  
        triton_time += time.time() - start_time

        start_time = time.time()
        _ = torch.cumsum(A, dim=-1)
        torch_time += time.time() - start_time

        del A
    
    A = torch.randn(b, n_kv, m, n, device='cuda', dtype=dtype)
    A_cs = efficient_cumsum(A)
    A_cs_ref = torch.cumsum(A, dim=-1)

    print(f"Triton kernel time: {triton_time}")
    print(f"Pytorch kernel time: {torch_time}")

    print("Avg error:", (A_cs - A_cs_ref).abs().mean())
    print("Max error:", (A_cs - A_cs_ref).abs().max())

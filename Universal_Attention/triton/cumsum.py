import triton
import triton.language as tl
import torch
import time
import math

# Phase 1: compute total sum of each chunk
@triton.jit
def _chunk_sum_kernel(input_ptr,      # pointer to input[0…L)
                      chunk_sums_ptr, # pointer to chunk_sums[0…num_chunks)
                      L: tl.constexpr,      # total length
                      chunk_size: tl.constexpr,  # size of each chunk
                      stride: tl.constexpr  # element stride (usually 1)
                      ):
    chunk_id = tl.program_id(0)
    start = chunk_id * chunk_size
    # actual elements in this chunk
    n = tl.minimum(chunk_size, L - start)
    sum_val = tl.float32(0.0)
    # simple serial reduction over up to chunk_size elements
    for i in range(0, n):
        sum_val += tl.load(input_ptr + (start + i) * stride)
    tl.store(chunk_sums_ptr + chunk_id, sum_val)

# Phase 3: do the chunked cumsum with tl.cumsum + add offset
@triton.jit
def _chunked_cumsum_kernel(input_ptr,
                           output_ptr,
                           chunk_offsets_ptr,  # exclusive-scan of chunk_sums
                           L: tl.constexpr,
                           chunk_size: tl.constexpr,
                           stride: tl.constexpr):
    chunk_id = tl.program_id(0)
    start = chunk_id * chunk_size
    n = tl.minimum(chunk_size, L - start)

    # load up to chunk_size elements (pad with zero)
    offs = start + tl.arange(0, chunk_size)
    x = tl.load(input_ptr + offs * stride, mask=offs < L, other=0.0)

    # 1) local inclusive scan across the chunk
    y = tl.cumsum(x, axis=0)

    # 2) add the precomputed offset for this chunk
    base = tl.load(chunk_offsets_ptr + chunk_id)
    y = y + base

    # 3) write back only the real elements
    tl.store(output_ptr + offs * stride, y, mask=offs < L)

# Host‐side driver to glue it all together
def chunked_cumsum_triton(x: torch.Tensor, chunk_size: int):
    """
    Compute y = cumsum(x) in chunks of size `chunk_size` using Triton.
    """
    assert x.ndim == 1, "1D example; extend to 2D by flattening rows."
    L = x.shape[0]
    num_chunks = math.ceil(L / chunk_size)

    # buffers on device
    chunk_sums   = torch.empty((num_chunks,), device=x.device, dtype=torch.float32)
    chunk_offsets = torch.empty_like(chunk_sums)
    y = torch.empty_like(x)

    # Phase 1: compute each chunk's total
    _chunk_sum_kernel[(num_chunks,)](
        x, chunk_sums,
        L, chunk_size, 1,
        num_warps=4,
    )

    # Phase 2: exclusive scan on host
    #   offsets[0] = 0, offsets[i] = sum(chunk_sums[:i])
    cs = chunk_sums.cpu().numpy()
    offs = cs.cumsum()
    chunk_offsets_cpu = torch.from_numpy(
        np.concatenate(([0.0], offs[:-1]), axis=0)
    )
    chunk_offsets.copy_(chunk_offsets_cpu.to(x.device))

    # Phase 3: chunked cumsum + add offsets
    _chunked_cumsum_kernel[(num_chunks,)](
        x, y, chunk_offsets,
        L, chunk_size, 1,
        num_warps=4,
    )
    return y

if __name__ == "__main__":
    x = torch.arange(1, 10, device='cuda', dtype=torch.float32)  # [1…9]
    y = chunked_cumsum_triton(x, chunk_size=3)
    # should be [1,2,3,4,5,6,7,8,9]
    print(y)  
    # verify
    assert torch.allclose(y, x.cumsum(dim=0))

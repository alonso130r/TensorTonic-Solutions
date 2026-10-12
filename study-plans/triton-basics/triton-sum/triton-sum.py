import torch
import triton
import triton.language as tl


@triton.jit
def sum_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = BLOCK_SIZE * pid 
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n 

    x = tl.load(x_ptr + offsets, mask=mask)

    tl.atomic_add(out_ptr + 0, tl.sum(x), sem="relaxed")

def solve(x: torch.Tensor, out: torch.Tensor) -> None:
    """Launch sum_kernel on the provided tensors."""
    n = x.numel()
    out.zero_()
    BLOCK_SIZE = 1024
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    sum_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK_SIZE)
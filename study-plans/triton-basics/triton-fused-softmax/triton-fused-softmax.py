import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(x_ptr, out_ptr, x_row_stride, out_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    row_start = x_ptr + x_row_stride * pid 
    out_start = out_ptr + out_row_stride * pid
    offsets = tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_cols 
    
    x = tl.load(row_start + offsets, mask=mask, other=-float("inf"))

    x -= tl.max(x, axis=0)
    numerator = tl.exp(x)
    denominator = tl.sum(numerator, axis=0)

    tl.store(out_start + offsets, numerator / denominator, mask=mask)

def solve(x: torch.Tensor, out: torch.Tensor) -> None:
    """Launch softmax_kernel with one program per row."""
    M, N = x.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (M,)
    softmax_kernel[grid](
        x, out, x.stride(0), out.stride(0), N, BLOCK_SIZE=BLOCK_SIZE,
    )
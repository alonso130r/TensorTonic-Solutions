import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_autotune_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_i = tl.program_id(axis=0)
    pid_j = tl.program_id(axis=1)

    offsets_am = (pid_i * BLOCK_M) + tl.arange(0, BLOCK_M)
    offsets_bn = (pid_j * BLOCK_N) + tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a_offsets = offsets_am[:, None] * stride_am + (k + offsets_k[None, :]) * stride_ak
        b_offsets = (k + offsets_k[:, None]) * stride_bk + offsets_bn[None, :] * stride_bn
        a_mask = (offsets_am[:, None] < M) & ((k + offsets_k[None, :]) < K)
        b_mask = ((k + offsets_k[:, None]) < K) & (offsets_bn[None, :] < N)
        
        a = tl.load(a_ptr + a_offsets, mask=a_mask, other=0.0)
        b = tl.load(b_ptr + b_offsets, mask=b_mask, other=0.0)

        acc = tl.dot(a, b, acc, input_precision="ieee")

    
    c_ptr_transf = c_ptr + (offsets_am[:, None] * stride_cm + offsets_bn[None, :] * stride_cn)
    c_mask = (offsets_am[:, None] < M) & (offsets_bn[None, :] < N)

    acc = acc.to(c_ptr.type.element_ty)

    tl.store(c_ptr_transf, acc, mask=c_mask)


def solve(A: torch.Tensor, B: torch.Tensor, out: torch.Tensor) -> None:
    """Launch the autotuned matmul kernel: out = A @ B."""
    M, K = A.shape
    K2, N = B.shape
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']), triton.cdiv(N, META['BLOCK_N']))
    matmul_autotune_kernel[grid](
        A, B, out,
        M=M, N=N, K=K,
        stride_am=A.stride(0), stride_ak=A.stride(1),
        stride_bk=B.stride(0), stride_bn=B.stride(1),
        stride_cm=out.stride(0), stride_cn=out.stride(1),
    )
"""
Questions to resolve:
1. There doesn't seem to be GEMM+bias fusion happening :(
"""

import torchinductor
import torchinductor.config
from torch.fx.experimental.proxy_tensor import make_fx

from torchinductor.compile_fx import compile_fx_inner
import torch
import torch.fx as fx

# torchinductor.config.triton.use_mm = True
torchinductor.config.triton.use_cutlass = True
torchinductor.config.debug = True

def f(a, b, bias, x):
    c = torch.mm(a, b)
    d = c + bias
    y = d * x
    return (c, d, y)

inps = [
    torch.empty(3, 4, device='cuda', requires_grad=True),  # a
    torch.empty(4, 5, device='cuda', requires_grad=True),  # b
    torch.empty(5, device='cuda', requires_grad=True),  # bias
    torch.empty(3, 5, device='cuda', requires_grad=True),  # x
]

new_mod = compile_fx_inner(make_fx(f)(*inps), inps)
new_mod(*inps)

"""
Output:

???
"""

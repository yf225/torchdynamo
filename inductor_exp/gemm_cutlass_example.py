import torchinductor
import torchinductor.config
from torch.fx.experimental.proxy_tensor import make_fx

from torchinductor.compile_fx import compile_fx_inner
import torch
import torch.fx as fx

# torchinductor.config.triton.use_mm = True
torchinductor.config.triton.use_cutlass = True
torchinductor.config.debug = True

def f(a, b):
    c = torch.mm(a, b)
    return (c,)

inps = [
    torch.empty(3, 4, device='cuda', requires_grad=True),  # a
    torch.empty(4, 5, device='cuda', requires_grad=True),  # b
]

new_mod = compile_fx_inner(make_fx(f)(*inps), inps)
print(new_mod(*inps))

"""
Output:

???
"""

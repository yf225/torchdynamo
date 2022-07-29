import torchinductor
import torchinductor.config
from torch.fx.experimental.proxy_tensor import make_fx

from torchinductor.compile_fx import compile_fx_inner
import torch
import torch.fx as fx

torchinductor.config.triton.use_mm = True
torchinductor.config.debug = True

def f(a, b, c, d):
    x1 = torch.mm(a, b)
    x2 = torch.mm(c, d)
    return (x1, x2)

inps = [
    torch.empty(3, 4, device='cuda', requires_grad=True),  # a
    torch.empty(4, 5, device='cuda', requires_grad=True),  # b
    torch.empty(3, 4, device='cuda', requires_grad=True),  # c
    torch.empty(4, 5, device='cuda', requires_grad=True),  # d
]

new_mod = compile_fx_inner(make_fx(f)(*inps), inps)
new_mod(*inps)

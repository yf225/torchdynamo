# Source: https://fb.quip.com/qqEdANmqIxCJ

import torchinductor
import torchinductor.config
from torch.fx.experimental.proxy_tensor import make_fx

from torchinductor.compile_fx import compile_fx_inner
import torch
import torch.fx as fx

torchinductor.config.debug = True

def f(a, b, bias):
    c = torch.mm(a, b)
    d = c + bias
    return (c, d)

inps = [
    torch.empty(3, 4, device='cuda', requires_grad=True),  # a
    torch.empty(4, 5, device='cuda', requires_grad=True),  # b
    torch.empty(5, device='cuda', requires_grad=True),  # bias
]

new_mod = compile_fx_inner(make_fx(f)(*inps), inps)
new_mod(*inps)

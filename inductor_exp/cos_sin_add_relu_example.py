import torchinductor
import torchinductor.config
from torch.fx.experimental.proxy_tensor import make_fx

from torchinductor.compile_fx import compile_fx_inner
import torch
import torch.fx as fx

torchinductor.config.debug = True

def f(x, y):
    x1 = x.cos()
    x2 = y.sin()
    x3 = torch.relu(x2)
    return (x1 + x3,)

inps = [torch.randn(2**5, device='cuda', requires_grad=True) for _ in range(2)]

new_mod = compile_fx_inner(make_fx(f)(*inps), inps)
new_mod(*inps)

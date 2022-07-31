import torch
import torchinductor
from torch.fx.experimental.proxy_tensor import make_fx
from torchinductor.compile_fx import compile_fx_inner


# torchinductor.config.triton.use_mm = True
torchinductor.config.triton.use_cutlass = True
torchinductor.config.triton.cudagraphs = True
torchinductor.config.debug = True

def f(a, b, x):
    c = torch.mm(a, b)
    y = b * x
    return (c, y)

tensor_A = torch.arange(3*4, device='cuda', requires_grad=True, dtype=torch.float).view(3, 4)
tensor_B = torch.arange(4*5, device='cuda', requires_grad=True, dtype=torch.float).view(4, 5)
tensor_X = torch.arange(4*5, device='cuda', requires_grad=True, dtype=torch.float).view(4, 5)
inps = [tensor_A, tensor_B, tensor_X]

new_mod = compile_fx_inner(make_fx(f)(*inps), inps)

out = new_mod(*inps)

print(out)

assert torch.allclose(out[0], tensor_A @ tensor_B)
assert torch.allclose(out[1], tensor_B * tensor_X)

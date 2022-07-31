import torchinductor
import torchinductor.config
from torch.fx.experimental.proxy_tensor import make_fx

from torchinductor.compile_fx import compile_fx_inner
import torch
import torch.fx as fx

# torchinductor.config.triton.use_mm = True
torchinductor.config.triton.use_cutlass = True
torchinductor.config.triton.cudagraphs = True
torchinductor.config.debug = True

def f(a, b, c, x):
    o1 = torch.mm(a, b) + c
    o2 = b * x
    return (o1, o2)

tensor_A = torch.arange(3*4, device='cuda', requires_grad=True, dtype=torch.float).view(3, 4)
tensor_B = torch.arange(4*5, device='cuda', requires_grad=True, dtype=torch.float).view(4, 5)
tensor_C = torch.arange(3*5, device='cuda', requires_grad=True, dtype=torch.float).view(3, 5)
tensor_X = torch.arange(4*5, device='cuda', requires_grad=True, dtype=torch.float).view(4, 5)
inps = [tensor_A, tensor_B, tensor_C, tensor_X]

new_mod = compile_fx_inner(make_fx(f)(*inps), inps)

out = new_mod(*inps)

print(out)

ref_out = f(*inps)

for t1, t2 in zip(out, ref_out):
    assert torch.allclose(t1, t2)

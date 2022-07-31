import torchinductor
import torchinductor.config
from torch.fx.experimental.proxy_tensor import make_fx

from torchinductor.compile_fx import compile_fx_inner
import torch
import torch.fx as fx

torchinductor.config.triton.use_mm = True
# torchinductor.config.triton.use_cutlass = True
torchinductor.config.debug = True

def f(a, b):
    c = torch.mm(a, b)
    return (c,)

tensor_A = torch.arange(3*4, device='cuda', requires_grad=True, dtype=torch.float).view(3, 4)
tensor_B = torch.arange(4*5, device='cuda', requires_grad=True, dtype=torch.float).view(4, 5)
inps = [tensor_A, tensor_B]

new_mod = compile_fx_inner(make_fx(f)(*inps), inps)
out = new_mod(*inps)[0]

torch.cuda.synchronize()
print(out)

pt_output = tensor_A @ tensor_B
assert torch.allclose(out, pt_output)

"""
Output:

???
"""

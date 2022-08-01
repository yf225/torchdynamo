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

# TODO: this doesn't work now, the output is slightly wrong
"""
out: (tensor([[ 70.,  77.,  84.,  91.,  98.],
        [191., 214., 237., 260., 278.],
        [312., 351., 390., 424., 462.]], device='cuda:0'), tensor([[  0.,   1.,   4.,   9.,  16.],
        [ 25.,  36.,  49.,  64.,  81.],
        [100., 121., 144., 169., 196.],
        [225., 256., 289., 324., 361.]], device='cuda:0'))
ref_out: (tensor([[ 70.,  77.,  84.,  91.,  98.],
        [190., 213., 236., 259., 282.],
        [310., 349., 388., 427., 466.]], device='cuda:0',
       grad_fn=<AddBackward0>), tensor([[  0.,   1.,   4.,   9.,  16.],
        [ 25.,  36.,  49.,  64.,  81.],
        [100., 121., 144., 169., 196.],
        [225., 256., 289., 324., 361.]], device='cuda:0',
       grad_fn=<MulBackward0>))
torch.mm(tensor_A, tensor_B): tensor([[ 70.,  76.,  82.,  88.,  94.],
        [190., 212., 234., 256., 278.],
        [310., 348., 386., 424., 462.]], device='cuda:0',
       grad_fn=<MmBackward0>)

Seems to be accessing illegal memory.
"""

def f(a, b, c, x):
    # TODO: figure out how to extract alpha and beta
    o1 = torch.mm(a, b) + c
    o2 = b * x
    return (o1, o2)

tensor_A = torch.arange(3*4, device='cuda', requires_grad=True, dtype=torch.float).view(3, 4)
tensor_B = torch.arange(4*5, device='cuda', requires_grad=True, dtype=torch.float).view(4, 5)
tensor_C = torch.arange(5, device='cuda', requires_grad=True, dtype=torch.float)
tensor_X = torch.arange(4*5, device='cuda', requires_grad=True, dtype=torch.float).view(4, 5)
inps = [tensor_A, tensor_B, tensor_C, tensor_X]

new_mod = compile_fx_inner(make_fx(f)(*inps), inps)

out = new_mod(*inps)

print(f"out: {out}")

ref_out = f(*inps)

print(f"ref_out: {ref_out}")

print(f"torch.mm(tensor_A, tensor_B): {torch.mm(tensor_A, tensor_B)}")

for t1, t2 in zip(out, ref_out):
    assert torch.allclose(t1, t2)

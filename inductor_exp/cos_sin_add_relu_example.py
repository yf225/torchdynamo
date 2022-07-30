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

"""
Output:

INFO torchinductor.codegen.triton: codegen (s0, 1)
INFO torchinductor.scheduler: NEW KERNEL
INFO torchinductor.scheduler: RUN buf0
INFO torchinductor.scheduler: blocked names: {}
INFO torchinductor.scheduler: blocked deps: {}
INFO torchinductor.scheduler: new fusable_deps: {MemoryDep(name='buf0', index=c0, size=(s0,))}

from ctypes import c_void_p, c_long
import torch
import random
from torch import empty_strided, as_strided
from torchinductor.codecache import CppCodeCache, TritonCodeCache

aten = torch.ops.aten

import triton
import triton.language as tl

from torchinductor.triton_ops.autotune import pointwise_heuristics
from torchinductor.triton_ops.autotune import reduction_heuristics
from torchinductor.triton_ops.autotune import grid

@pointwise_heuristics(size_hints=[32])
@triton.jit
def kernel0(in_ptr0, in_ptr1, out_ptr0, ks0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp2 = tl.load(in_ptr1 + x0, xmask)
    tmp1 = tl.cos(tmp0)
    tmp3 = tl.sin(tmp2)
    tmp4 = tl.maximum(0, tmp3)
    tmp5 = tmp1 + tmp4
    tl.store(out_ptr0 + x0, tmp5, xmask)


def call(x_1, y_1):
    x_1_size = x_1.size()
    s0 = x_1_size[0]
    buf0 = empty_strided((s0, ), (1, ), device='cuda', dtype=torch.float32)
    kernel0[grid(s0)](x_1, y_1, buf0, s0, s0)
    return (buf0, )


if __name__ == "__main__":
    from torchdynamo.testing import rand_strided
    from benchmarks.microbenchmarks.microbench import print_performance
    x_1 = rand_strided((32, ), (1, ), device='cuda', dtype=torch.float32)
    y_1 = rand_strided((32, ), (1, ), device='cuda', dtype=torch.float32)
    print_performance(lambda: call(x_1, y_1))

PyCodeCache /tmp/torchinductor_willfeng/h6/ch6yoxqwj562yj43k5x7kptpndeajkloebhqnjqrm4j3wp6pucop.py
"""

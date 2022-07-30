import torchinductor
import torchinductor.config
from torch.fx.experimental.proxy_tensor import make_fx

from torchinductor.compile_fx import compile_fx_inner
import torch
import torch.fx as fx

torchinductor.config.triton.use_mm = True
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

INFO torchinductor.scheduler: RUN EXTERN buf0
INFO torchinductor.scheduler: blocked names: {MemoryDep(name='buf0', index=c0, size=(s0*s2,)): [SchedulerNodeBox(value=SchedulerNode(name='buf1'))], MemoryDep(name='buf1', index=c0, size=(s0*s2,)): [SchedulerNodeBox(value=SchedulerNode(name='buf2'))]}
INFO torchinductor.scheduler: blocked deps: {'buf0': [SchedulerNodeBox(value=SchedulerNode(name='buf1'))], 'buf1': [SchedulerNodeBox(value=SchedulerNode(name='buf2'))]}
INFO torchinductor.scheduler: new fusable_deps: set()
INFO torchinductor.codegen.triton: codegen (s0*s2, 1)
INFO torchinductor.scheduler: NEW KERNEL
INFO torchinductor.scheduler: RUN buf1
INFO torchinductor.codegen.triton: REPLACED s2*x1 + x0 with x2 (sizes {x1: s0, x0: s2} with {x2: s0*s2})
INFO torchinductor.codegen.triton: REPLACED s2*x1 + x0 with x2 (sizes {x1: s0, x0: s2} with {x2: s0*s2})
INFO torchinductor.scheduler: RUN buf2
INFO torchinductor.scheduler: blocked names: {}
INFO torchinductor.scheduler: blocked deps: {}
INFO torchinductor.scheduler: new fusable_deps: {MemoryDep(name='buf2', index=c0, size=(s0*s2,)), MemoryDep(name='buf1', index=c0, size=(s0*s2,))}

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
from torchinductor.triton_ops.matmul import matmul_out as triton_mm_out

@pointwise_heuristics(size_hints=[16])
@triton.jit
def kernel0(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, ks0, ks1, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % ks1
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask)
    tmp3 = tl.load(in_ptr2 + x2, xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tl.store(out_ptr0 + x2, tmp2, xmask)
    tl.store(out_ptr1 + x2, tmp4, xmask)


def call(a_1, b_1, bias_1, x_1):
    a_1_size = a_1.size()
    s0 = a_1_size[0]
    s1 = a_1_size[1]
    b_1_size = b_1.size()
    s2 = b_1_size[1]
    buf0 = empty_strided((s0, s2), (s2, 1), device='cuda', dtype=torch.float32)
    triton_mm_out(a_1, b_1, out=buf0)
    buf1 = empty_strided((s0, s2), (s2, 1), device='cuda', dtype=torch.float32)
    buf2 = empty_strided((s0, s2), (s2, 1), device='cuda', dtype=torch.float32)
    kernel0_xnumel = s0*s2
    kernel0[grid(kernel0_xnumel)](buf0, bias_1, x_1, buf1, buf2, s0, s2, kernel0_xnumel)
    return (buf0, buf1, buf2, )


if __name__ == "__main__":
    from torchdynamo.testing import rand_strided
    from benchmarks.microbenchmarks.microbench import print_performance
    a_1 = rand_strided((3, 4), (4, 1), device='cuda', dtype=torch.float32)
    b_1 = rand_strided((4, 5), (5, 1), device='cuda', dtype=torch.float32)
    bias_1 = rand_strided((5, ), (1, ), device='cuda', dtype=torch.float32)
    x_1 = rand_strided((3, 5), (5, 1), device='cuda', dtype=torch.float32)
    print_performance(lambda: call(a_1, b_1, bias_1, x_1))

PyCodeCache /tmp/torchinductor_willfeng/kq/ckqvodqhoal3ftvvelcc5o2awqhrzfnrzfgse5rhlbc34t5lhesp.py
"""

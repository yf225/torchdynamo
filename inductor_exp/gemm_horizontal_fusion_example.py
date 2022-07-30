"""
Questions to resolve:
1. There doesn't seem to be horizontal fusion between the two GEMMs :(
"""

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

"""
Output:

INFO torchinductor.scheduler: RUN EXTERN buf0
INFO torchinductor.scheduler: blocked names: {}
INFO torchinductor.scheduler: blocked deps: {}
INFO torchinductor.scheduler: new fusable_deps: set()
INFO torchinductor.scheduler: RUN EXTERN buf1
INFO torchinductor.scheduler: blocked names: {}
INFO torchinductor.scheduler: blocked deps: {}
INFO torchinductor.scheduler: new fusable_deps: set()

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


def call(a_1, b_1, c_1, d_1):
    a_1_size = a_1.size()
    s0 = a_1_size[0]
    s1 = a_1_size[1]
    b_1_size = b_1.size()
    s2 = b_1_size[1]
    buf0 = empty_strided((s0, s2), (s2, 1), device='cuda', dtype=torch.float32)
    triton_mm_out(a_1, b_1, out=buf0)
    buf1 = empty_strided((s0, s2), (s2, 1), device='cuda', dtype=torch.float32)
    triton_mm_out(c_1, d_1, out=buf1)
    return (buf0, buf1, )


if __name__ == "__main__":
    from torchdynamo.testing import rand_strided
    from benchmarks.microbenchmarks.microbench import print_performance
    a_1 = rand_strided((3, 4), (4, 1), device='cuda', dtype=torch.float32)
    b_1 = rand_strided((4, 5), (5, 1), device='cuda', dtype=torch.float32)
    c_1 = rand_strided((3, 4), (4, 1), device='cuda', dtype=torch.float32)
    d_1 = rand_strided((4, 5), (5, 1), device='cuda', dtype=torch.float32)
    print_performance(lambda: call(a_1, b_1, c_1, d_1))

PyCodeCache /tmp/torchinductor_willfeng/wg/cwgodxncd3lxjw3caim5xcgcztsxps2ll5g7mn7dpurpkchqtynm.py
"""
import torch
import torchinductor
from torch.fx.experimental.proxy_tensor import make_fx
from torchinductor.compile_fx import compile_fx_inner


# torchinductor.config.triton.use_mm = True
torchinductor.config.triton.use_cutlass = True
torchinductor.config.triton.cudagraphs = True
torchinductor.config.debug = True

def f(a, b):
    c = torch.mm(a, b)
    return (c,)

tensor_A = torch.arange(3*4, device='cuda', requires_grad=True, dtype=torch.float).view(3, 4)
tensor_B = torch.arange(4*5, device='cuda', requires_grad=True, dtype=torch.float).view(4, 5)
inps = [tensor_A, tensor_B]

new_mod = compile_fx_inner(make_fx(f)(*inps), inps)

out = new_mod(*inps)[0]

print(out)

pt_output = tensor_A @ tensor_B
assert torch.allclose(out, pt_output)

"""
Output:

INFO torchinductor.scheduler: RUN EXTERN buf0
INFO torchinductor.scheduler: NEW KERNEL
INFO torchinductor.scheduler: blocked names: {}
INFO torchinductor.scheduler: blocked deps: {}
INFO torchinductor.scheduler: new fusable_deps: {MemoryDep(name='buf0', index=c0, size=(s0*s2,))}

from ctypes import c_void_p, c_long
import torch
import random
from torch import empty_strided, as_strided
from torchdynamo.testing import rand_strided
from torchinductor.codecache import CppCodeCache, TritonCodeCache

aten = torch.ops.aten

import triton
import triton.language as tl

from torchinductor.triton_ops.autotune import pointwise_heuristics
from torchinductor.triton_ops.autotune import reduction_heuristics
from torchinductor.triton_ops.autotune import grid
import torch
from torchinductor.compile_fx import stream_dict

# System modules
import numpy as np
import os
import os.path
import sys
import ctypes
from types import SimpleNamespace

# CUDA Python modules
from cuda import cuda
from cuda import nvrtc

# CUTLASS modules
import library
import manifest as cutlass_manifest
import generator
import rt

username = os.environ.get('USER')

gemm = None

# TODO(yf225): add autotuning and kernel selection

def compile_cuda_module():
    global gemm

    if gemm is not None:
        return

    cuda_ver = "11.4"
    cuda_arch = "80"  # assuming A100

    # Construct an SGEMM
    manifest = cutlass_manifest.Manifest()
    generator.GenerateSM50_Simt(manifest, cuda_ver)

    # Look up the GEMM operation
    operation = manifest.operations_by_name['cutlass_simt_sgemm_128x128_8x2_nn_align1']

    # Construct a runtime GEMM operation
    gemm = rt.Gemm(operation)

    # Construct a module
    architectures = [80,]
    include_paths = [
      f'/fsx/users/{username}/cutlass/include',
      f'/fsx/users/{username}/cutlass/tools/util/include',
      f'/usr/local/cuda-{cuda_ver}/include',
      f'/usr/local/cuda-{cuda_ver}/targets/x86_64-linux/include',
    ]

    compilation_options = rt.CompilationOptions(architectures, include_paths)
    # NOTE: compilation needs to happen outside of CUDA stream capture (e.g. when capturing for CUDA graph)
    module = rt.Module('module.cu', [gemm], compilation_options)


# # TODO(yf225): Can we use TorchDynamo / TorchInductor caching mechanism to reuse kernel for subsequent runs?
# @cached
def kernel0(
    tensor_A_torch,
    tensor_B_torch,
    tensor_D_torch,
    M,
    N,
    K,
):
    global gemm
    global stream_dict

    if "cutlass_stream" not in stream_dict:
        # If currently CUTLASS kernel is not required to use any pre-existing stream,
        # then create a new stream dedicated to CUTLASS which is also exposed publicly
        # for easier control.
        stream_dict["cutlass_stream"] = torch.cuda.Stream()
    cuda_stream_ptr = stream_dict["cutlass_stream"].cuda_stream

    # Formula: D = alpha * (A @ B) + beta * C
    # TODO(yf225): for some weird reason, no matter what bias tensor we pass into the following GEMM operation, it will not be used. We need to debug this.
    tensor_C_torch = torch.empty(M, N, device='cuda', dtype=tensor_A_torch.dtype)

    arguments = rt.GemmArguments()
    arguments.problem_size = rt.GemmCoord(M, N, K)
    arguments.A = rt.TensorRef(tensor_A_torch.data_ptr(), tensor_A_torch.stride()[0])
    arguments.B = rt.TensorRef(tensor_B_torch.data_ptr(), tensor_B_torch.stride()[0])
    arguments.C = rt.TensorRef(tensor_C_torch.data_ptr(), tensor_C_torch.stride()[0])
    arguments.D = rt.TensorRef(tensor_D_torch.data_ptr(), tensor_D_torch.stride()[0])

    host_workspace = bytearray(gemm.get_host_workspace_size(arguments))
    device_workspace = None

    launch_config = gemm.plan(arguments)
    byte_count = gemm.initialize(host_workspace, device_workspace, launch_config, arguments, stream=cuda_stream_ptr)
    err = gemm.run(host_workspace, device_workspace, launch_config, stream=cuda_stream_ptr)
    if err != cuda.CUresult.CUDA_SUCCESS:
      raise RuntimeError('CUDA Error %s' % str(err))

compile_cuda_module()


def call(a_1, b_1):
    a_1_size = a_1.size()
    s0 = a_1_size[0]
    s1 = a_1_size[1]
    b_1_size = b_1.size()
    s2 = b_1_size[1]
    buf0 = rand_strided((s0, s2), (s2, 1), device='cuda', dtype=torch.float32)
    kernel0(a_1, b_1, buf0, s0, s2, s1, )
    return (buf0, )


if __name__ == "__main__":
    from torchdynamo.testing import rand_strided
    from benchmarks.microbenchmarks.microbench import print_performance
    a_1 = rand_strided((3, 4), (4, 1), device='cuda', dtype=torch.float32)
    b_1 = rand_strided((4, 5), (5, 1), device='cuda', dtype=torch.float32)
    print_performance(lambda: call(a_1, b_1))

PyCodeCache /tmp/torchinductor_willfeng/vl/cvlqklwn5ygv7owzadjj2zzqg3glx326epy264o55afsuajq5zgn.py
tensor([[ 70.,  76.,  82.,  88.,  94.],
        [190., 212., 234., 256., 278.],
        [310., 348., 386., 424., 462.]], device='cuda:0')
"""

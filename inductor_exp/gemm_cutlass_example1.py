"""
Questions to resolve:
1. What CUTLASS code should we generate? Ideally it's autotuning configs + populated kernel template, and then upon first run we find the best tile sizes etc.
e.g.

```
generated_kernel = jinja2.Template("""
# ... The generated GEMM epilogue fusion kernel ...
""")

@hpcfuser.cached  # Can we use TorchDynamo / TorchInductor caching mechanism to reuse kernel for subsequent runs?
def call_generated_kernel(ins, outs):
    # TODO(yf225): this requires just-in-time complication, see if can reuse CUTLASS-Python infra.
    return hpcfuser.autotune(configs=[...])(generated_kernel)(*ins)
```

TODO Next step:
1. Generate the actual GEMM+bias kernel with provided GEMM info and bias info
"""

import torchinductor
import torchinductor.config
from torch.fx.experimental.proxy_tensor import make_fx

from torchinductor.compile_fx import compile_fx_inner
import torch
import torch.fx as fx

# torchinductor.config.triton.use_mm = True
torchinductor.config.triton.use_cutlass = True
torchinductor.config.debug = True

def f(a, b, bias, mul):
    c = torch.mm(a, b)
    d = c + bias
    e = b * mul
    return (c, d, e)

inps = [
    torch.empty(3, 4, device='cuda', requires_grad=True),  # a
    torch.empty(4, 5, device='cuda', requires_grad=True),  # b
    torch.empty(5, device='cuda', requires_grad=True),  # bias
    torch.empty(4, 5, device='cuda', requires_grad=True),  # e
]

new_mod = compile_fx_inner(make_fx(f)(*inps), inps)
new_mod(*inps)

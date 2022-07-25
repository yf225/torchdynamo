# add some debug printouts
debug = False

# dead code elimination
dce = False

# assume there will be no backwards
forward_only = False

# assume input tensors are dynamic
dynamic_shapes = True

# assume weight tensors are fixed size
static_weight_shapes = True

# enable some approximation algorithms
approximations = False

# put correctness assertions in generated code
size_asserts = True

# enable loop reordering based on input orders
pick_loop_orders = True

# generate inplace computations
inplace_buffers = False

# codegen benchmark harness
benchmark_harness = True

# control store vs recompute heuristic
realize_reads_threshold = 4
realize_bytes_threshold = 2000

# fallback to eager for random/dropout, this is slow but useful for debugging
fallback_random = False

# python_key_normalize versus aot_autograd
aot_autograd = True


# config specific to codegen/cpp.pp
class cpp:
    threads = -1  # set to cpu_count()
    simdlen = None
    min_chunk_size = 4096
    cxx = (
        None,  # download gcc12 from conda-forge if conda is installed
        "g++-12",
        "g++-11",
        "g++-10",
        "clang++-12",
        "clang++-11",
        "clang++-10",
        "g++",
    )


# config specific to codegen/triton.py
class triton:

    # Use cudagraphs on output code
    cudagraphs = True

    # Monkey patching to lower overheads
    hackery = False

    # choose conv backend, "aten" or "triton" or "autotune"
    convolution = "aten"

    # Always load full blocks (rather than broadcasting inside the block)
    # Set default as True because otherwise will encouter `map::at` error
    # in triton if loading from 1-dim tensor using 2-dim pointer offset
    # https://triton-lang.slack.com/archives/C01L1FLTX70/p1656023403343639
    # could be set as False if triton fixes the bug later
    dense_indexing = True if convolution != "aten" else False

    # limit tiling dimensions
    # Disable tiling until we figure out how tiling and fusion work together
    max_tiles = 2
    tile_broadcasting = False

    # put each kernel in its own file
    many_files = False

    # use triton.autotune?
    autotune = True

    # enable codegen to use Triton's mm
    use_mm = False

    use_bmm = False

import torch

def try_gpu(x: torch.Tensor | torch.nn.Module) -> torch.Tensor | torch.nn.Module:
    if torch.cuda.is_available():
        return x.cuda()
    return x


def get_texts() -> list[str]:
    return [
        "torch.compile is the latest method to speed up your PyTorch code!",
        "torch.compile makes PyTorch code run faster by JIT-compiling PyTorch code into optimized kernels, all while requiring minimal code changes.",
        "In this tutorial, we cover basic torch.compile usage, and demonstrate the advantages of torch.compile over previous PyTorch compiler solutions, such as TorchScript and FX Tracing.",
        "NOTE: a modern NVIDIA GPU (H100, A100, or V100) is recommended for this tutorial in order to reproduce the speedup numbers shown below and documented elsewhere.",
        "torch.compile is included in the latest PyTorch.",
        "Running TorchInductor on GPU requires Triton, which is included with the PyTorch 2.0 nightly binary.",
        "If Triton is still missing, try installing torchtriton via pip (pip install torchtriton --extra-index-url \"https://download.pytorch.org/whl/nightly/cu117\" for CUDA 11.7).",
        "Arbitrary Python functions can be optimized by passing the callable to torch.compile. We can then call the returned optimized function in place of the original function.",
        "Alternatively, we can decorate the function.",
        "We can also optimize torch.nn.Module instances.",
        "Nested function calls within the decorated function will also be compiled.",
        "In the same fashion, when compiling a module all sub-modules and methods within it, that are not in a skip list, are also compiled.",
        "We can also disable some functions from being compiled by using torch.compiler.disable.",
        "Suppose you want to disable the tracing on just the complex_function function, but want to continue the tracing back in complex_conjugate.",
        "In this case, you can use torch.compiler.disable(recursive=False) option. Otherwise, the default is recursive=True.",
    ]

import torch
import intel_extension_for_pytorch as ipex
import importlib

class CondFunc:
    def __new__(cls, orig_func, sub_func, cond_func):
        self = super(CondFunc, cls).__new__(cls)
        if isinstance(orig_func, str):
            func_path = orig_func.split('.')
            for i in range(len(func_path)-1, -1, -1):
                try:
                    resolved_obj = importlib.import_module('.'.join(func_path[:i]))
                    break
                except ImportError:
                    pass
            for attr_name in func_path[i:-1]:
                resolved_obj = getattr(resolved_obj, attr_name)
            orig_func = getattr(resolved_obj, func_path[-1])
            setattr(resolved_obj, func_path[-1], lambda *args, **kwargs: self(*args, **kwargs))
        self.__init__(orig_func, sub_func, cond_func)
        return lambda *args, **kwargs: self(*args, **kwargs)
    def __init__(self, orig_func, sub_func, cond_func):
        self.__orig_func = orig_func
        self.__sub_func = sub_func
        self.__cond_func = cond_func
    def __call__(self, *args, **kwargs):
        if not self.__cond_func or self.__cond_func(self.__orig_func, *args, **kwargs):
            return self.__sub_func(self.__orig_func, *args, **kwargs)
        else:
            return self.__orig_func(*args, **kwargs)

def check_device(device):
    return bool((isinstance(device, torch.device) and device.type == "cuda") or (isinstance(device, str) and "cuda" in device) or isinstance(device, int))

def return_xpu(device):
    return f"xpu:{device[-1]}" if isinstance(device, str) and ":" in device else f"xpu:{device}" if isinstance(device, int) else torch.device("xpu") if isinstance(device, torch.device) else "xpu"

def ipex_no_cuda(orig_func, *args, **kwargs):
    torch.cuda.is_available = lambda: False
    orig_func(*args, **kwargs)
    torch.cuda.is_available = torch.xpu.is_available

original_autocast = torch.autocast
def ipex_autocast(*args, **kwargs):
    if args[0] == "cuda":
        return original_autocast("xpu", *args[1:], **kwargs)
    else:
        return original_autocast(*args, **kwargs)

original_torch_cat = torch.cat
def torch_cat(input, *args, **kwargs):
    if len(input) == 3 and (input[0].dtype != input[1].dtype or input[2].dtype != input[1].dtype):
        return original_torch_cat([input[0].to(input[1].dtype), input[1], input[2].to(input[1].dtype)], *args, **kwargs)
    else:
        return original_torch_cat(input, *args, **kwargs)

original_interpolate = torch.nn.functional.interpolate
def interpolate(input, size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None, antialias=False):
    if antialias or align_corners is not None:
        return_device = input.device
        return_dtype = input.dtype
        return original_interpolate(input.to("cpu", dtype=torch.float32), size=size, scale_factor=scale_factor, mode=mode,
        align_corners=align_corners, recompute_scale_factor=recompute_scale_factor, antialias=antialias).to(return_device, dtype=return_dtype)
    else:
        return original_interpolate(input, size=size, scale_factor=scale_factor, mode=mode,
        align_corners=align_corners, recompute_scale_factor=recompute_scale_factor, antialias=antialias)

original_linalg_solve = torch.linalg.solve
def linalg_solve(orig_func, A, B, *args, **kwargs):
    if A.device != torch.device("cpu") or B.device != torch.device("cpu"):
        return_device = A.device
        original_linalg_solve(A.to("cpu"), B.to("cpu"), *args, **kwargs).to(return_device)
    else:
        original_linalg_solve(A, B, *args, **kwargs)

def ipex_hijacks():
    CondFunc('torch.Tensor.to',
        lambda orig_func, self, device=None, *args, **kwargs: orig_func(self, return_xpu(device), *args, **kwargs),
        lambda orig_func, self, device=None, *args, **kwargs: check_device(device))
    CondFunc('torch.Tensor.cuda',
        lambda orig_func, self, device=None, *args, **kwargs: orig_func(self, return_xpu(device), *args, **kwargs),
        lambda orig_func, self, device=None, *args, **kwargs: check_device(device))
    CondFunc('torch.empty',
        lambda orig_func, *args, device=None, **kwargs: orig_func(*args, device=return_xpu(device), **kwargs),
        lambda orig_func, *args, device=None, **kwargs: check_device(device))
    CondFunc('torch.load',
        lambda orig_func, *args, map_location=None, **kwargs: orig_func(*args, return_xpu(map_location), **kwargs),
        lambda orig_func, *args, map_location=None, **kwargs: map_location is None or check_device(map_location))
    CondFunc('torch.randn',
        lambda orig_func, *args, device=None, **kwargs: orig_func(*args, device=return_xpu(device), **kwargs),
        lambda orig_func, *args, device=None, **kwargs: check_device(device))
    CondFunc('torch.ones',
        lambda orig_func, *args, device=None, **kwargs: orig_func(*args, device=return_xpu(device), **kwargs),
        lambda orig_func, *args, device=None, **kwargs: check_device(device))
    CondFunc('torch.zeros',
        lambda orig_func, *args, device=None, **kwargs: orig_func(*args, device=return_xpu(device), **kwargs),
        lambda orig_func, *args, device=None, **kwargs: check_device(device))
    CondFunc('torch.tensor',
        lambda orig_func, *args, device=None, **kwargs: orig_func(*args, device=return_xpu(device), **kwargs),
        lambda orig_func, *args, device=None, **kwargs: check_device(device))
    CondFunc('torch.linspace',
        lambda orig_func, *args, device=None, **kwargs: orig_func(*args, device=return_xpu(device), **kwargs),
        lambda orig_func, *args, device=None, **kwargs: check_device(device))

    CondFunc('torch.Generator',
        lambda orig_func, device=None: torch.xpu.Generator(device),
        lambda orig_func, device=None: device is not None and device != torch.device("cpu") and device != "cpu")

    CondFunc('torch.batch_norm',
        lambda orig_func, input, weight, bias, *args, **kwargs: orig_func(input,
        weight if weight is not None else torch.ones(input.size()[1], device=input.device),
        bias if bias is not None else torch.zeros(input.size()[1], device=input.device), *args, **kwargs),
        lambda orig_func, input, *args, **kwargs: input.device != torch.device("cpu"))
    CondFunc('torch.instance_norm',
        lambda orig_func, input, weight, bias, *args, **kwargs: orig_func(input,
        weight if weight is not None else torch.ones(input.size()[1], device=input.device),
        bias if bias is not None else torch.zeros(input.size()[1], device=input.device), *args, **kwargs),
        lambda orig_func, input, *args, **kwargs: input.device != torch.device("cpu"))

    #Functions with dtype errors:
    CondFunc('torch.nn.modules.GroupNorm.forward',
        lambda orig_func, self, input: orig_func(self, input.to(self.weight.data.dtype)),
        lambda orig_func, self, input: input.dtype != self.weight.data.dtype)
    CondFunc('torch.nn.modules.linear.Linear.forward',
        lambda orig_func, self, input: orig_func(self, input.to(self.weight.data.dtype)),
        lambda orig_func, self, input: input.dtype != self.weight.data.dtype)
    CondFunc('torch.bmm',
        lambda orig_func, input, mat2, *args, **kwargs: orig_func(input, mat2.to(input.dtype), *args, **kwargs),
        lambda orig_func, input, mat2, *args, **kwargs: input.dtype != mat2.dtype)
    CondFunc('torch.nn.functional.layer_norm',
        lambda orig_func, input, normalized_shape=None, weight=None, *args, **kwargs:
        orig_func(input.to(weight.data.dtype), normalized_shape, weight, *args, **kwargs),
        lambda orig_func, input, normalized_shape=None, weight=None, *args, **kwargs:
        weight is not None and input.dtype != weight.data.dtype)

    #Diffusers Float64 (ARC GPUs doesn't support double or Float64):
    if not torch.xpu.has_fp64_dtype():
        CondFunc('torch.from_numpy',
        lambda orig_func, ndarray: orig_func(ndarray.astype('float32')),
        lambda orig_func, ndarray: ndarray.dtype == float)

    #Broken functions when torch.cuda.is_available is True:
    CondFunc('torch.utils.data.dataloader._BaseDataLoaderIter.__init__',
        lambda orig_func, *args, **kwargs: ipex_no_cuda(orig_func, *args, **kwargs),
        lambda orig_func, *args, **kwargs: True)

    #Functions that make compile mad with CondFunc:
    torch.autocast = ipex_autocast
    torch.cat = torch_cat
    torch.linalg.solve = linalg_solve
    torch.nn.functional.interpolate = interpolate

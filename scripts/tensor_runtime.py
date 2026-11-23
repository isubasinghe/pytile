"""Execute static float32 tensor PTX plus its compiler-generated JSON manifest."""

import ctypes as ct
import json
import math
from pathlib import Path

import numpy as np
from run_gpu import Driver


def fingerprint(data):
    value = 0xCBF29CE484222325
    for byte in data:
        value = ((value ^ byte) * 0x100000001B3) & ((1 << 64) - 1)
    return f"{value:016x}"


def read_manifest(ptx):
    ptx = Path(ptx)
    image = ptx.read_bytes()
    manifest = json.loads(ptx.with_suffix(".json").read_text())
    if manifest.get("version") != 1 or manifest.get("dtype") != "float32":
        raise ValueError("Unsupported tensor manifest version or dtype")
    if manifest.get("ptx_fnv1a64") != fingerprint(image):
        raise ValueError("PTX and manifest do not match; recompile the PTX artifact")
    shapes = manifest.get("buffers", [])
    if not shapes or any(
        not shape or any(type(d) is not int or d <= 0 for d in shape)
        or math.prod(shape) > (2**63 - 1) // 4 for shape in shapes
    ):
        raise ValueError("Invalid tensor buffer shapes")

    def buffer_id(value):
        return type(value) is int and 0 <= value < len(shapes)

    if not all(buffer_id(i) for i in manifest.get("inputs", [])):
        raise ValueError("Invalid input buffer ID")
    if not buffer_id(manifest.get("output")):
        raise ValueError("Invalid output buffer ID")
    for kernel in manifest.get("kernels", []):
        if not isinstance(kernel.get("name"), str) or not kernel["name"]:
            raise ValueError("Invalid kernel name")
        if not kernel.get("buffers") or not all(buffer_id(i) for i in kernel["buffers"]):
            raise ValueError("Invalid kernel buffer IDs")
        for key in ("grid", "block"):
            dims = kernel.get(key, [])
            if len(dims) != 3 or any(type(d) is not int or d <= 0 for d in dims):
                raise ValueError("Invalid kernel launch dimensions")
        if kernel["block"] != [128, 1, 1] or kernel["grid"][1:] != [1, 1]:
            raise ValueError("Unsupported kernel launch layout")
        if kernel["grid"][0] > 2**31 - 1:
            raise ValueError("Kernel grid is too large")
    if not manifest.get("kernels"):
        raise ValueError("Tensor manifest contains no kernels")
    return image, manifest


class TensorProgram:
    """A reusable callable owning a private CUDA context and its device buffers.

    Use as a context manager. Arguments must be float32 arrays with the exact
    compiled shapes. Calls are synchronous and intended for one host thread.
    """

    def __init__(self, ptx, device_index=0):
        image, self.manifest = read_manifest(ptx)
        self.input_shapes = [tuple(self.manifest["buffers"][i])
                             for i in self.manifest["inputs"]]
        self.output_shape = tuple(self.manifest["buffers"][self.manifest["output"]])
        self.driver = Driver()
        self.context = ct.c_void_p()
        self.module = ct.c_void_p()
        self.buffers = []
        self.kernels = []
        self.driver.call("cuInit", 0)
        previous = ct.c_void_p()
        self.driver.call("cuCtxGetCurrent", ct.byref(previous))
        device = ct.c_int()
        self.driver.call("cuDeviceGet", ct.byref(device), device_index)
        try:
            self.driver.call("cuCtxCreate_v2", ct.byref(self.context), 0, device)
            data = ct.create_string_buffer(image)
            self.driver.call("cuModuleLoadData", ct.byref(self.module), data)
            for shape in self.manifest["buffers"]:
                pointer = ct.c_uint64()
                self.driver.call("cuMemAlloc_v2", ct.byref(pointer), math.prod(shape) * 4)
                self.buffers.append(pointer)
            for spec in self.manifest["kernels"]:
                function = ct.c_void_p()
                self.driver.call("cuModuleGetFunction", ct.byref(function), self.module,
                                 spec["name"].encode())
                self.kernels.append((function, spec))
        except Exception:
            self.close()
            raise
        finally:
            self.driver.call("cuCtxSetCurrent", previous)

    def __call__(self, *arguments):
        if not self.context.value:
            raise RuntimeError("TensorProgram is closed")
        if len(arguments) != len(self.input_shapes):
            raise ValueError(f"Expected {len(self.input_shapes)} inputs, got {len(arguments)}")
        arrays = []
        for index, (argument, shape) in enumerate(zip(arguments, self.input_shapes, strict=True)):
            array = np.asarray(argument)
            if array.dtype != np.float32 or array.shape != shape:
                raise ValueError(f"Input {index}: expected float32 {shape}, "
                                 f"got {array.dtype} {array.shape}")
            arrays.append(np.ascontiguousarray(array))
        previous = ct.c_void_p()
        self.driver.call("cuCtxGetCurrent", ct.byref(previous))
        self.driver.call("cuCtxSetCurrent", self.context)
        try:
            for array, index in zip(arrays, self.manifest["inputs"], strict=True):
                self.driver.call("cuMemcpyHtoD_v2", self.buffers[index],
                                 array.ctypes.data, array.nbytes)
            # The default CUDA stream runs these dependent kernels in order.
            for function, spec in self.kernels:
                pointer_array = ct.c_void_p * len(spec["buffers"])
                pointers = pointer_array(
                    *[ct.addressof(self.buffers[i]) for i in spec["buffers"]]
                )
                self.driver.call("cuLaunchKernel", function, *spec["grid"], *spec["block"],
                                 0, None, pointers, None)
            self.driver.call("cuCtxSynchronize")
            result = np.empty(self.output_shape, dtype=np.float32)
            self.driver.call("cuMemcpyDtoH_v2", result.ctypes.data,
                             self.buffers[self.manifest["output"]], result.nbytes)
            return result
        finally:
            self.driver.call("cuCtxSetCurrent", previous)

    def close(self):
        if self.context.value:
            # Destroying the private context frees every allocation and module,
            # including partially constructed instances after a CUDA error.
            self.driver.lib.cuCtxDestroy_v2(self.context)
            self.context = ct.c_void_p()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

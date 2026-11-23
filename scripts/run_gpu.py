"""Run the reference kernel via the CUDA Driver API; no CUDA toolkit needed.

Fixed contract: compute(float *x, float *y, float *out), 128 float32 elements,
one block of 128 threads. Change this contract when extending the compiler.
"""

import argparse
import ctypes as ct
import os
from pathlib import Path

COUNT = 128


class Driver:
    def __init__(self):
        override = os.environ.get("PYTILE_CUDA_DRIVER")
        candidates = [override] if override else [
            "libcuda.so.1", "/run/opengl-driver/lib/libcuda.so.1"
        ]
        errors = []
        for candidate in candidates:
            try:
                self.lib = ct.CDLL(candidate)
                break
            except OSError as error:
                errors.append(str(error))
        else:
            raise RuntimeError("Cannot load the NVIDIA driver: " + "; ".join(errors))
        ptr, uint, size, devptr = ct.c_void_p, ct.c_uint, ct.c_size_t, ct.c_uint64
        signatures = {
            "cuInit": [uint],
            "cuDeviceGet": [ct.POINTER(ct.c_int), ct.c_int],
            "cuCtxCreate_v2": [ct.POINTER(ptr), uint, ct.c_int],
            "cuCtxDestroy_v2": [ptr],
            "cuCtxSynchronize": [],
            "cuCtxGetCurrent": [ct.POINTER(ptr)],
            "cuCtxSetCurrent": [ptr],
            "cuModuleLoadData": [ct.POINTER(ptr), ptr],
            "cuModuleGetFunction": [ct.POINTER(ptr), ptr, ct.c_char_p],
            "cuModuleUnload": [ptr],
            "cuMemAlloc_v2": [ct.POINTER(devptr), size],
            "cuMemFree_v2": [devptr],
            "cuMemcpyHtoD_v2": [devptr, ptr, size],
            "cuMemcpyDtoH_v2": [ptr, devptr, size],
            "cuLaunchKernel": [ptr, uint, uint, uint, uint, uint, uint, uint,
                               ptr, ct.POINTER(ptr), ct.POINTER(ptr)],
            "cuGetErrorString": [ct.c_int, ct.POINTER(ct.c_char_p)],
        }
        for name, args in signatures.items():
            function = getattr(self.lib, name)
            function.argtypes = args
            function.restype = ct.c_int

    def call(self, name, *args):
        result = getattr(self.lib, name)(*args)
        if result:
            message = ct.c_char_p()
            self.lib.cuGetErrorString(result, ct.byref(message))
            detail = message.value.decode() if message.value else "unknown CUDA error"
            raise RuntimeError(f"{name}: {detail} ({result})")


def run(ptx: Path, device_index: int = 0):
    import numpy as np

    driver = Driver()
    driver.call("cuInit", 0)
    device = ct.c_int()
    driver.call("cuDeviceGet", ct.byref(device), device_index)
    context, module, kernel = ct.c_void_p(), ct.c_void_p(), ct.c_void_p()
    driver.call("cuCtxCreate_v2", ct.byref(context), 0, device)
    allocations = []
    try:
        image = ct.create_string_buffer(ptx.read_bytes())
        driver.call("cuModuleLoadData", ct.byref(module), image)
        driver.call("cuModuleGetFunction", ct.byref(kernel), module, b"compute")
        for _ in range(3):
            address = ct.c_uint64()
            driver.call("cuMemAlloc_v2", ct.byref(address), COUNT * 4)
            allocations.append(address)
        arguments = (ct.c_void_p * 3)(*[ct.addressof(p) for p in allocations])
        rng = np.random.default_rng(42)
        x = rng.uniform(-10, 10, COUNT).astype(np.float32)
        y = rng.uniform(-10, 10, COUNT).astype(np.float32)
        cases = [(x, y), (x, -x), (np.zeros_like(x), y)]
        for a, b in cases:
            output = np.full(COUNT, np.nan, dtype=np.float32)
            for address, array in zip(allocations, (a, b, output), strict=True):
                driver.call("cuMemcpyHtoD_v2", address, array.ctypes.data, array.nbytes)
            driver.call("cuLaunchKernel", kernel, 1, 1, 1, COUNT, 1, 1, 0, None, arguments, None)
            driver.call("cuCtxSynchronize")
            driver.call("cuMemcpyDtoH_v2", output.ctypes.data, allocations[2], output.nbytes)
            expected = (a + b) * (a + b)
            np.testing.assert_allclose(output, expected, rtol=1e-6, atol=1e-6)
        print(f"GPU verification passed: {len(cases)} cases × {COUNT} float32 values")
    finally:
        # Destroying this private context also releases resources if an earlier
        # CUDA call failed. Best-effort cleanup preserves the original error.
        for address in allocations:
            driver.lib.cuMemFree_v2(address)
        if module.value:
            driver.lib.cuModuleUnload(module)
        driver.lib.cuCtxDestroy_v2(context)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ptx", type=Path, nargs="?", default=Path("build/smoke.ptx"))
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    try:
        run(args.ptx, args.device)
    except (OSError, RuntimeError, AssertionError) as error:
        parser.exit(1, f"GPU check failed: {error}\n")

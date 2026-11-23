"""Launch a scalar i64 kernel with one thread using the CUDA Driver API.

ABI: the provided i64 arguments, followed by a device pointer to one i64 result.
Example: python3 scripts/run_scalar_gpu.py build/add.ptx --args 2 3 --expect 5
"""

import argparse
import ctypes as ct
from pathlib import Path

from run_gpu import Driver


def i64(text):
    value = int(text)
    if not -(1 << 63) <= value < (1 << 63):
        raise argparse.ArgumentTypeError("argument must fit in a signed 64-bit integer")
    return value


def run(ptx, values, kernel_name="compute_kernel", device_index=0):
    driver = Driver()
    driver.call("cuInit", 0)
    device = ct.c_int()
    driver.call("cuDeviceGet", ct.byref(device), device_index)
    context, module, kernel = ct.c_void_p(), ct.c_void_p(), ct.c_void_p()
    driver.call("cuCtxCreate_v2", ct.byref(context), 0, device)
    output = ct.c_uint64()
    try:
        image = ct.create_string_buffer(ptx.read_bytes())
        driver.call("cuModuleLoadData", ct.byref(module), image)
        driver.call("cuModuleGetFunction", ct.byref(kernel), module, kernel_name.encode())
        driver.call("cuMemAlloc_v2", ct.byref(output), ct.sizeof(ct.c_int64))
        arguments = [ct.c_int64(value) for value in values] + [output]
        pointers = (ct.c_void_p * len(arguments))(
            *[ct.addressof(argument) for argument in arguments]
        )
        driver.call("cuLaunchKernel", kernel, 1, 1, 1, 1, 1, 1, 0, None, pointers, None)
        driver.call("cuCtxSynchronize")
        result = ct.c_int64()
        driver.call("cuMemcpyDtoH_v2", ct.byref(result), output, ct.sizeof(result))
        return result.value
    finally:
        if output.value:
            driver.lib.cuMemFree_v2(output)
        if module.value:
            driver.lib.cuModuleUnload(module)
        driver.lib.cuCtxDestroy_v2(context)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ptx", type=Path)
    parser.add_argument("--args", type=i64, nargs="*", required=True)
    parser.add_argument("--kernel", default="compute_kernel")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--expect", type=i64)
    args = parser.parse_args()
    try:
        result = run(args.ptx, args.args, args.kernel, args.device)
        if args.expect is not None and result != args.expect:
            raise RuntimeError(f"expected {args.expect}, got {result}")
        print(result)
    except (OSError, RuntimeError) as error:
        parser.exit(1, f"GPU execution failed: {error}\n")

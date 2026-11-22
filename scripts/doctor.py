"""Report the Rust compiler toolchain and GPU access separately."""

import shutil
import subprocess

for name in ("rustc", "cargo", "mlir-opt", "mlir-translate", "llc", "python3", "nvidia-smi"):
    print(f"{name:22} {shutil.which(name) or 'not found'}")

if shutil.which("nvidia-smi"):
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,compute_cap", "--format=csv,noheader"],
        capture_output=True, text=True,
    )
    print(result.stdout or result.stderr or f"nvidia-smi exited {result.returncode}")
    if result.returncode:
        print("If running in an agent sandbox, repeat this check in a normal terminal.")

print("Rust compiler -> Python AST -> GPU MLIR -> NVVM -> PTX sm_75.")
print("Python/NumPy are used only by the support scripts, not by the compiler.")

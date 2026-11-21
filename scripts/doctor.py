"""Report compilation tools separately from GPU execution prerequisites."""

import shutil
import subprocess

for name in ("python3", "cuda-tile-opt", "cuda-tile-translate", "tileiras", "nvidia-smi"):
    print(f"{name:22} {shutil.which(name) or 'not found'}")

if shutil.which("nvidia-smi"):
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,compute_cap", "--format=csv,noheader"],
        capture_output=True, text=True,
    )
    print(result.stdout or result.stderr or f"nvidia-smi exited {result.returncode}")

print("Tile IR validation and serialization do not require a GPU.")
print("GPU execution requires a supported GPU, tileiras/toolkit, and working driver.")

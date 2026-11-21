# PyTile

A one-day compiler project: parse a small Python subset and emit CUDA Tile IR.

```python
def compute(x, y):
    t = x + y
    return t * t
```

The intended path is Python AST → CUDA Tile MLIR → Tile IR bytecode → GPU binary.
The frontend is deliberately an exercise: AST inspection works; validation and
IR generation are marked in `pytile/frontend.py` for you to implement.

## Start

```sh
cd ~/Projects/pytile
nix develop .#frontend  # Python, NumPy, pytest, ruff; no LLVM build needed
make ast
make test
```

Build the pinned CUDA Tile tools, then enter the full shell:

```sh
nix build --cores 8 .
nix develop
make smoke
make doctor
```

`flake.lock` pins NVIDIA's sources, LLVM, and Nix dependencies. The LLVM revision
matches NVIDIA's `cmake/IncludeLLVM.cmake`. When updating cuda-tile, check that
revision too. The build uses textual MLIR, so Python MLIR bindings are unnecessary.
The initial source build may take considerable time and disk space. Subsequent
shell entries reuse the built tools. No CUDA driver is needed for this build.

## Your implementation

1. In `pytile/frontend.py`, validate one function, two array arguments, and a return.
2. Track argument/local names and create a fresh SSA value for each operation.
3. Emit a kernel that loads two 128-element float32 tiles, adds them, and stores
   the result. Initially support exactly 128 elements and one launched tile.
4. Add assignments and multiplication, keeping temporary values inside the kernel.
5. Reject unsupported syntax with source locations. Add acceptance tests for
   unknown variables, unsupported calls, and valid expression chains.
6. Verify emitted MLIR with `cuda-tile-opt` and serialize it as in `make smoke`.
7. On a supported GPU machine, adapt NVIDIA's host example, run the kernel, and
   compare against NumPy. Extend to multiple tiles only after one tile works.

```sh
python3 -m pytile examples/add.py --dump-ast
# Once you implement compile_module:
python3 -m pytile examples/add.py -o build/add.mlir
```

Keep the first version to equal-sized contiguous float32 arrays, `+`, `*`, local
assignments, and a final return. Defer loops, broadcasting, arbitrary calls,
autodiff, custom optimisation passes, and partial tiles.

## Machine limitations

This machine has an RTX 2070 SUPER (Turing, compute capability 7.5) with a working
NVIDIA 595.71.05 driver. It supports ordinary CUDA. An initial `nvidia-smi` failure
was caused by the restricted agent environment; the check succeeds outside it.
Turing is outside CUDA Tile's documented supported architectures. This project
can compile and validate Tile IR here; Tile IR execution needs another supported
machine. A conventional CUDA/PTX backend could run on this GPU. No driver or
operating-system configuration is changed by this project. `tileiras` and a GPU
launcher are not included in the local development shell.

## References

- [CUDA Tile IR, tools, and host launch example](https://github.com/NVIDIA/cuda-tile)
- [CUDA Tile specification](https://docs.nvidia.com/cuda/tile-ir/)
- [Existing cuTile Python frontend and hardware prerequisites](https://github.com/NVIDIA/cutile-python)
- [Python AST reference](https://docs.python.org/3/library/ast.html)

# PyTile

PyTile is a Rust learning compiler for a Python subset. It uses RustPython to
parse Python and Melior to build MLIR, an intermediate representation.

Python-to-MLIR and tensor-to-GPU lowering remain exercises. The existing GPU
backend converts standalone `gpu.module` input to NVIDIA PTX.

## Start

Run these commands from the repository root:

```sh
nix develop
make ast           # Show the Python syntax tree
make ir            # Write the MLIR builder example to build/high_level.mlir
make verify-ir     # Check that example with mlir-opt
make test          # Run Rust tests
make test-backend  # Check the backend with external tools
make lint          # Check Rust and Python code
make gpu-smoke     # Build and run the GPU reference kernel
```

`make ir` builds `(x + y) * (x + y)` on `tensor<128xf32>` values directly.
It does not compile Python.

The Nix shell supplies native dependencies. Melior 0.26.6 requires the project's
LLVM/MLIR 21.1.8. Update them together. Use `cargo check --locked` for compiler
diagnostics without an executable build.

## Develop the compiler

- [src/frontend.rs](src/frontend.rs) parses Python, records argument types, and contains the unfinished MLIR lowering code.
- [examples/ir_builder.rs](examples/ir_builder.rs) demonstrates Melior builders and MLIR checks.
- [src/ir.rs](src/ir.rs) creates the MLIR context.
- [src/main.rs](src/main.rs) runs the frontend, checks MLIR, and writes output.
- [src/backend.rs](src/backend.rs) converts GPU MLIR to PTX with external tools.

Start with two tensor arguments and `return x + y`. Then add assignments and
multiplication.

The `compile` command enters the unfinished lowering code:

```sh
cargo run --locked -- compile examples/add.py --args '(2, 3)' -o build/add.mlir
```

`--args` evaluates a Python tuple with Rumpy available as `np`. The compiler
does not execute the source module or target function. Supply every argument
positionally, including defaults. Use `--function forward` when the source
defines multiple functions.

`CompileUnit` records scalar types and array dtype and shape without copying
tensor data. Run the metadata example:

```sh
cargo run --locked --example runtime_types
```

## GPU reference

`make gpu-smoke` compiles [examples/smoke.mlir](examples/smoke.mlir) and checks
its output against NumPy. The kernel computes `(x + y) * (x + y)` for 128 elements.

The backend targets `sm_75` / PTX 7.5 for the RTX 2070 SUPER. The check requires
an accessible NVIDIA driver. It does not require a separate CUDA toolkit.
Set `PYTILE_CUDA_DRIVER` to override the driver library path.

## MNIST example

[examples/MNIST.md](examples/MNIST.md) describes a NumPy `784 → 32 → 10` network.
Run `make mnist-demo` for synthetic data or `make mnist-test` for numerical checks.
The compiler does not yet lower this network to MLIR.

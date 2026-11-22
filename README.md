# PyTile

A compiler written in **Rust**, accepting a subset of **Python** and using
**Melior**, the Rust bindings to MLIR's C API, for its intermediate representation.

```text
Python → RustPython AST → Melior / MLIR func+arith tensor module
                                      ↓ future lowering
                           GPU MLIR → NVVM → LLVM IR → PTX → GPU
```

There are no custom IR classes or handwritten MLIR printer. The example uses
Melior's `Context`, `Module`, `Block`, `Region`, types, and dialect builders.
MLIR itself verifies and prints the resulting module.

## Start

```sh
cd ~/Projects/pytile
nix develop
make ast           # inspect Python syntax using RustPython's parser
make ir            # construct and print real MLIR through Melior
make verify-ir     # also check the printed result with mlir-opt
make test          # Rust tests, native MLIR tests, and the builder example test
make test-backend  # external-tool GPU backend integration checks
make lint
make gpu-smoke     # existing handwritten GPU reference → PTX → GPU
```

`make ir` writes `build/high_level.mlir`. Its example constructs `(x + y) *
(x + y)` on whole `tensor<128xf32>` values using `func.func`, `arith.addf`,
`arith.mulf`, and `func.return`. It does not compile Python yet.

## Where to work

- `src/frontend.rs`: your Python validation and lowering exercise. Return a
  `melior::ir::Module<'c>` tied to the supplied `&'c melior::Context`.
- `examples/ir_builder.rs`: a complete example using native Melior builders,
  including an in-process verification and printer/parser round-trip test.
- `src/ir.rs`: only creates a context and loads MLIR's dialects.
- `src/main.rs`: owns the context, invokes your frontend, verifies the module,
  and prints it through MLIR.
- `src/backend.rs`: the existing external-tool GPU-MLIR-to-PTX pipeline.

The `compile` command evaluates sample arguments, creates a typed `CompileUnit`,
and enters `CompileUnit::compile_module`:

```sh
cargo run --locked -- compile examples/add.py --args '(2, 3)' -o build/add.mlir
```

This reaches your argument lowering code now, then reports that MLIR lowering
is not implemented. It will write the output only once you return verified MLIR.
`--args` is a Python tuple expression evaluated with Rumpy available as `np`;
the source module and target function body are not executed. For tensor inputs,
use `--args '(np.zeros(128, dtype=np.float32), np.ones(128, dtype=np.float32))'`.
Use `--function forward` to select a function when the source defines several;
otherwise a single top-level function is selected automatically.

Start with two tensor arguments and `return x + y`, then add assignments and
multiplication. Map Python names to Melior SSA values while constructing the
function block. The caller keeps the context alive while the module and its
values are in use. Use `module.as_operation().verify()` to check the IR.

Once you implement Python lowering:

```sh
cargo run --locked -- compile examples/square_sum.py --args '(np.zeros(128, dtype=np.float32), np.ones(128, dtype=np.float32))' -o build/generated.mlir
mlir-opt build/generated.mlir -o build/generated.verified.mlir
```

Python-to-MLIR lowering deliberately remains your exercise.
The parser accepts more Python than the intended subset, so validate supported
constructs explicitly. High-level tensor-to-GPU lowering is also future work.

## Runtime argument types

`CompileUnit::from_runtime_args(function, arguments, vm)` binds actual RustPython
objects to the parsed function's parameter names. Rumpy arrays contribute their
numeric dtype and shape without copying tensor data. Python bool, int, and float
arguments are distinguished; Python integers remain arbitrary-precision metadata,
not an assumed MLIR `i64`. Each call creates a separate compilation unit.

```rust
let mut unit = CompileUnit::from_runtime_args(function, arguments, vm)?;
let x_type = unit.variable_type("x");
// Once you implement lowering, this entry point uses the same populated unit:
let mlir = unit.compile_module(&context, &module)?;
```

Run the working metadata example with:

```sh
cargo run --locked --example runtime_types
```

It inspects scalar and array inputs for `add.py`, then the five float32 inputs
for `mnist_forward.py`. The selected Python function body is not executed.
`emit_variable` reads the supplied type from `CompileUnit::variables`; creating
MLIR block arguments and inferring intermediate types remain the lowering exercise.

Currently every parameter must be supplied positionally, including parameters
with defaults. Positional-only parameters work; keyword-only parameters, variadic
signatures, unsupported objects, and nonnumeric arrays produce errors. This API
does not yet install a Python decorator or automatically intercept calls.

## Development build speed

Development builds explicitly enable Cargo's incremental compilation. The Nix
shell selects Mold through its compiler wrapper to speed up executable linking.
Your own Rust code retains full debug information, debug assertions and overflow
checks. Third-party dependencies omit debug information, reducing build and link
work; inspecting their local variables in a debugger requires restoring those
symbols.

Re-enter `nix develop` after changing the shell. The first build with these
settings rebuilds affected dependencies; subsequent edits reuse them. Avoid
`cargo clean` during normal development because it discards that cache. Use
`cargo check --locked` when you only need compiler diagnostics and no executable.
Changing a Python input file does not require rebuilding the Rust compiler.

## Native dependencies

Melior is pinned to **0.26.6**, compatible with the project's **LLVM/MLIR 21.1.8**.
Newer Melior releases can require newer LLVM releases; update these together.
`Cargo.lock` pins the Rust dependencies and `flake.lock` pins Nix packages.

The Nix shell provides a combined LLVM/MLIR development prefix because Nix splits
headers, libraries, and executables into separate outputs. It configures
`MLIR_SYS_210_PREFIX`, `TABLEGEN_210_PREFIX`, bindgen/libclang, and LLVM's native
link dependencies. TableGen's C++ build uses optimization to work with Nix's
FORTIFY hardening. No hand-set environment variables should be necessary.

Both `nix develop` and `nix develop .#frontend` now include native MLIR development
dependencies, because the Rust compiler links against MLIR. The default shell
also provides Python/NumPy for support scripts. The compiler itself does not
invoke Python. The initial native binding build takes longer than later builds.

## Existing GPU backend

The `ptx` command still accepts a standalone `gpu.module`, not the new high-level
`func`/tensor module. Use the handwritten GPU example until you add that lowering:

```sh
cargo run --locked -- ptx examples/smoke.mlir -o build/smoke.ptx
python3 scripts/run_gpu.py build/smoke.ptx
```

The reference kernel takes three `memref<128xf32>` arguments (x, y, output) and
runs one block of 128 threads. The Python harness expects `(x + y) * (x + y)` and
compares GPU output against NumPy. It is a correctness check, not a benchmark.

The target is `sm_75` / PTX 7.5 for the RTX 2070 SUPER. The installed CUDA driver
compiles PTX at load time; no separate CUDA toolkit or nvcc is needed for this
kernel. The harness finds `libcuda.so.1` or NixOS's
`/run/opengl-driver/lib/libcuda.so.1`; `PYTILE_CUDA_DRIVER` overrides the path.
If an agent sandbox hides the GPU, run the GPU check in a normal terminal.

## References

- [Melior 0.26.6](https://docs.rs/crate/melior/0.26.6)
- [Melior source and examples](https://github.com/mlir-rs/melior)
- [RustPython parser](https://docs.rs/rustpython-parser/0.4.0/rustpython_parser/)
- [MLIR arith dialect](https://mlir.llvm.org/docs/Dialects/ArithOps/)
- [MLIR GPU dialect](https://mlir.llvm.org/docs/Dialects/GPU/)

## Neural-network example

See [examples/MNIST.md](examples/MNIST.md) for a NumPy `784 → 32 → 10` MLP with
one-hot digit labels, manual backpropagation, and SGD. `make mnist-demo` runs a
synthetic smoke demo; `make mnist-test` checks gradients and numerical stability.
`examples/mnist_forward.py` is the next compiler input to target after elementwise
arithmetic. Python-to-MLIR lowering of the network remains an exercise.

# PyTile

PyTile is a Rust learning compiler for a Python subset. It uses RustPython to
parse Python and Melior to build MLIR, an intermediate representation.

The frontend compiles scalar integer arithmetic, float32 matrix multiplication,
and matching-shape float32 tensor arithmetic to MLIR. The GPU
backend converts scalar `i64` functions and standalone `gpu.module` input to
NVIDIA PTX. Tensor-to-GPU lowering remains an exercise.

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

After scalar `square_sum`, use the [matrix scores example](examples/matrix_scores.py)
below as the next compiler goal.

The `compile` command emits MLIR for the supported subset:

```sh
cargo run --locked -- compile examples/add.py --args '(2, 3)' -o build/add.mlir
```

Compile that function to CUDA PTX and run it on the GPU:

```sh
cargo run --locked -- ptx build/add.mlir -o build/add.ptx
python3 scripts/run_scalar_gpu.py build/add.ptx --args 2 3 --expect 5
```

The second command prints `5`. The backend adds a `compute_kernel` entry point
that takes the scalar arguments followed by a device pointer to one `i64`
result. Launch it with exactly one block and one thread. For another function
name, pass `--kernel <name>_kernel` to the runner. The number and order of
`--args` must match the compiled function. Scalar GPU lowering currently accepts
one function with `i64` parameters and one `i64` result; arithmetic uses fixed
64-bit integers, not Python's arbitrary-precision integers.

The compiler's sample arguments select types; the runner supplies the actual
values. PTX compilation requires the Nix toolchain, while execution additionally
requires an accessible NVIDIA driver. No separate CUDA toolkit is needed.

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

## Next goal: a tiny matrix classifier

[examples/matrix_scores.py](examples/matrix_scores.py) computes two class scores
for each of three samples:

```python
def compute(x, weights, bias):
    scores = x @ weights
    return scores + bias
```

The weights are illustrative, not trained. Higher scores select the class;
these are not probabilities. Run `make matrix-demo` for the NumPy reference:

```text
[[ 3. -1.]
 [ 0.  2.]
 [ 2.  1.]]
Predicted classes: [0, 1, 0]
```

The frontend now supports checkpoints 1–3 below. Tensor execution on the GPU
is the next compiler goal:

1. **Matrix inputs and returns:** represent `float32` arrays as ranked MLIR
   types. Start with an identity function, `return x`, preserving shape `[3, 2]`.
2. **Matrix multiplication:** lower `x @ weights` with shapes `[3, 2] @ [2, 2]`
   to `[3, 2]`. Expected intermediate: `[[2, -1], [-1, 2], [1, 1]]`.
   Check the inner dimensions match; unlike addition, operand shapes need not
   be identical. If using `linalg.matmul`, initialize its accumulator to zero.
3. **Matrix addition:** add the `[3, 2]` bias elementwise and produce the scores
   above. This deliberately avoids broadcasting. Verify the emitted MLIR.
4. **GPU execution:** allocate array and intermediate buffers, lower the matrix
   operations to GPU kernels, and compare the returned scores with NumPy.
   The existing scalar GPU adapter/runner needs extension for arrays.

Compile the example to verified MLIR:

```sh
cargo run --locked -- compile examples/matrix_scores.py \
  --args '(
    np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float32),
    np.array([[2, -1], [-1, 2]], dtype=np.float32),
    np.array([[1, 0], [1, 0], [1, 0]], dtype=np.float32)
  )' -o build/matrix_scores.mlir
```

Keep `argmax` in the reference runner for now. Later goals are a vector bias
with broadcasting, ReLU, and a second dense layer—the pieces of the MNIST
forward pass.

## MNIST example

[examples/MNIST.md](examples/MNIST.md) describes a NumPy `784 → 32 → 10` network.
Run `make mnist-demo` for synthetic data or `make mnist-test` for numerical checks.
Use `make mnist-forward` for one inference batch. `examples/run_mnist.py` owns
the CLI; data loading and CPU training are separate from the model functions.
The compiler does not yet lower this network to MLIR.

# PyTile

PyTile is a Rust learning compiler for a Python subset. It uses RustPython to
parse Python and Melior to build MLIR, an intermediate representation.

The frontend compiles scalar arithmetic, float32 matrix multiplication,
broadcast tensor arithmetic, and `np.maximum` to MLIR. The GPU backend converts
scalar `i64` functions, supported static float32 tensor graphs, and standalone
`gpu.module` input to NVIDIA PTX.

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
- [src/backend.rs](src/backend.rs) lowers scalar/tensor functions to GPU MLIR and PTX.

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

The compiler now supports these checkpoints for static float32 shapes:

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
   The tensor backend emits a buffer/launch manifest for `TensorProgram` in
   `scripts/tensor_runtime.py`.

Compile the example to verified MLIR:

```sh
cargo run --locked -- compile examples/matrix_scores.py \
  --args '(
    np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float32),
    np.array([[2, -1], [-1, 2]], dtype=np.float32),
    np.array([[1, 0], [1, 0], [1, 0]], dtype=np.float32)
  )' -o build/matrix_scores.mlir
```

Keep `argmax` in the reference runner for now. Vector bias broadcasting, ReLU,
and a second dense layer are supported by the MNIST forward target below.

## MNIST example

[examples/mnist_forward.py](examples/mnist_forward.py) defines the `784 → 32 → 10`
network. Compile the full forward pass for a batch of 32:

```sh
make mnist-compile  # writes build/mnist_forward.mlir
```

This emits two `linalg.matmul` operations, broadcast bias additions, and ReLU.
Elementwise operations broadcast trailing dimensions when they match or one is
size one. Tensor dtypes must match; Python float scalars adopt float32 when used
with float32 tensors. Standalone Python floats use f64. `np.maximum` supports
two positional float operands, including scalar thresholds such as `0.0`.

Compile and execute the complete forward pass on an NVIDIA GPU:

```sh
make mnist-gpu
```

This regenerates the MLIR, compiles `build/mnist_forward.ptx` and its companion
`build/mnist_forward.json`, then checks GPU logits against NumPy on five cases.
The runner uses the existing `forward_fn` interface, including uploads, launches,
and copying the result back. It checks random, zero, negative-activation, and
noncontiguous inputs as well as the synthetic dataset batch.

To run an existing artifact, optionally using a local MNIST archive:

```sh
python3 scripts/run_mnist_gpu.py build/mnist_forward.ptx
python3 scripts/run_mnist_gpu.py build/mnist_forward.ptx --data build/mnist.npz
```

Keep the PTX and JSON together; the runtime checks that they match. The manifest
describes exact input shapes, intermediate buffers, and kernel launch order.
`TensorProgram` is a reusable callable that takes float32 arrays and returns a
NumPy array. Input shapes must match the compiled signature. A different batch
size requires recompilation.

The first tensor backend supports positive static dimensions, standard rank-2
matrix multiplication, fills, and the frontend's pointwise broadcast operations.
It launches one kernel per tensor operation (seven for this forward pass), with
one output element per thread and a serial dot product inside each matmul thread.
Kernel fusion and tiled matrix multiplication are future optimization work.
Training remains a CPU reference; these checks use initialized weights and test
numerical agreement, not trained digit-recognition accuracy.

Run `make mnist-demo` for synthetic data or `make mnist-test` for numerical checks.
Use `make mnist-forward` for one NumPy inference batch. `examples/run_mnist.py` owns
the CLI; data loading and CPU training are separate from the model functions.

"""Run compiled MNIST inference on CUDA and compare it with the NumPy model."""

import argparse
import sys
from pathlib import Path

import numpy as np
from tensor_runtime import TensorProgram

# Keep the model/data reference in examples and the CUDA runtime in scripts.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from mnist_cpu import initialize  # noqa: E402
from mnist_forward import forward  # noqa: E402
from run_mnist import main as run_reference  # noqa: E402


def run(ptx, device=0, seed=0, data=None):
    with TensorProgram(ptx, device) as program:
        shapes = program.input_shapes
        if (len(shapes) != 5 or len(shapes[0]) != 2 or shapes[0][1] != 784
                or shapes[1:] != [(784, 32), (32,), (32, 10), (10,)]
                or program.output_shape != (shapes[0][0], 10)):
            raise ValueError("Expected a compiled MNIST forward signature")
        batch = shapes[0][0]
        maximum_error = 0.0
        count = 0

        def checked(*arguments):
            nonlocal maximum_error, count
            actual = program(*arguments)
            expected = forward(*arguments)
            np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
            maximum_error = max(maximum_error, float(np.max(np.abs(actual - expected))))
            count += 1
            return actual

        options = ["--forward-only", "--batch-size", str(batch), "--seed", str(seed)]
        if data is not None:
            options += ["--data", str(data)]
        print(f"CUDA inference: {len(program.kernels)} kernels, batch {batch}")
        run_reference(options, forward_fn=checked)

        # Reuse the same device buffers with different data and weights, including
        # zero/negative activations; this catches stale buffers and launch tails.
        rng = np.random.default_rng(seed + 1)
        x = rng.normal(size=(batch, 784)).astype(np.float32)
        parameters = initialize(rng)
        checked(x, *parameters)
        checked(x[:, ::-1], *parameters)
        checked(np.zeros_like(x), *parameters)
        w1, b1, w2, b2 = parameters
        checked(x, np.zeros_like(w1), np.full_like(b1, -1), w2, b2)
        print(f"GPU verification passed: {count} cases × {batch * 10} logits; "
              f"max absolute error {maximum_error:.3g}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ptx", type=Path, nargs="?", default=Path("build/mnist_forward.ptx"))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data", type=Path, help="Optional local MNIST .npz archive")
    args = parser.parse_args()
    try:
        run(args.ptx, args.device, args.seed, args.data)
    except (OSError, ValueError, RuntimeError, AssertionError) as error:
        parser.exit(1, f"MNIST GPU check failed: {error}\n")

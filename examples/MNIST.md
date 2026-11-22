# MNIST MLP compiler example

The network is `784 → 32 → 10`: flatten a 28×28 grayscale image, apply a dense
layer with ReLU, then a dense layer producing ten class logits. Images contain
normalized pixel values, **not one-hot values**. Labels are one-hot vectors:
digit 3 becomes `[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]`.

## Files

- `mnist_forward.py`: small, independently runnable NumPy forward function. Start
  compiling this function before tackling the training code.
- `mnist_mlp.py`: full CPU reference with initialization, stable softmax and
  cross-entropy, manual backpropagation, minibatch SGD, and evaluation.
- `test_mnist_mlp.py`: finite-difference gradient checks, stability checks,
  learning on a small fixture, and MNIST archive preprocessing checks.

Only NumPy is required; it is already in `nix develop`.

## Run

From the project root:

```sh
nix develop
make mnist-demo   # synthetic noisy patterns, not handwritten digits
make mnist-test
```

For real MNIST, use a Keras-format `.npz` archive (Keras is not required). A verified
copy is cached at `build/mnist.npz` during setup. The `build` directory is ignored
by Git. The program never downloads data automatically.

```sh
python3 examples/mnist_mlp.py --data build/mnist.npz --epochs 3 --limit 2048
# Later, use all 60,000 training images:
python3 examples/mnist_mlp.py --data build/mnist.npz --epochs 5 --limit 60000
```

To obtain another copy, the archive is distributed at the
[TensorFlow/Keras MNIST archive](https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz).
Its SHA-256 is `731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1`,
as recorded in the [Keras loader](https://github.com/keras-team/keras/blob/master/keras/src/datasets/mnist.py).
The archive has `x_train`, `y_train`, `x_test`, and `y_test` arrays. Images are uint8
with shape `[N, 28, 28]`; labels are integers from 0 to 9.

The demo reports loss and accuracy before and after training. Synthetic results
are labelled separately and are not measurements of digit recognition. No
trained weights are saved. `--seed`, `--batch-size`, and `--learning-rate` control
the experiment; evaluation does not update the model.

## Compiler targets

Use a fixed batch size initially, such as 32:

| Value | Shape | Type |
|---|---|---|
| Input `x` | `[32, 784]` | float32 |
| First weights `w1` | `[784, 32]` | float32 |
| First bias `b1` | `[32]` | float32 |
| Hidden activations | `[32, 32]` | float32 |
| Second weights `w2` | `[32, 10]` | float32 |
| Second bias `b2` | `[10]` | float32 |
| Logits / one-hot targets | `[32, 10]` | float32 |

Suggested milestones for your Python-to-Melior lowering:

1. Function arguments, assignments, and a return with ranked tensor types.
2. Matrix multiplication (`@`) using `linalg.matmul` and a zero-initialized output.
3. Bias broadcasting using indexing maps in `linalg.generic`.
4. Elementwise ReLU using `arith.maximumf` inside an elementwise operation.
5. Softmax/cross-entropy: reductions along the class axis, `math.exp`, and `math.log`.
6. Backward computations and SGD, after inference matches the NumPy reference.

The current compiler **does not compile this network yet**. Its Python lowering
is still your exercise, and this example adds matrix multiplication, broadcasting,
NumPy-call handling, and reductions beyond the earlier elementwise examples.
`mnist_forward.py` is the small first target; imports and the data/training CLI
can stay outside the compiled function. Training here runs in NumPy on the CPU,
not through the existing fixed-size GPU smoke kernel.

To inspect its AST:

```sh
cargo run --locked -- ast examples/mnist_forward.py
```

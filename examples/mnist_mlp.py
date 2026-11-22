"""Train a 784 -> 32 -> 10 NumPy MLP with one-hot labels and manual SGD.

Run without arguments for synthetic smoke data. Pass --data /path/to/mnist.npz
for real MNIST (keys: x_train, y_train, x_test, y_test). Nothing is downloaded.
"""

import argparse
from pathlib import Path

import numpy as np
from mnist_forward import forward


def one_hot(labels, classes=10):
    labels = np.asarray(labels)
    if labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Labels must be a 1D integer array")
    if np.any(labels < 0) or np.any(labels >= classes):
        raise ValueError(f"Labels must be in [0, {classes})")
    return np.eye(classes, dtype=np.float32)[labels]


def initialize(rng, inputs=784, hidden=32, classes=10, dtype=np.float32):
    w1 = (rng.standard_normal((inputs, hidden)) * np.sqrt(2.0 / inputs)).astype(dtype)
    b1 = np.zeros(hidden, dtype=dtype)
    w2 = (rng.standard_normal((hidden, classes)) * np.sqrt(2.0 / hidden)).astype(dtype)
    b2 = np.zeros(classes, dtype=dtype)
    return w1, b1, w2, b2


def softmax(logits):
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def cross_entropy(logits, targets):
    # Work in log space rather than taking log(softmax), which can underflow.
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    log_probs = shifted - np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
    return -np.sum(targets * log_probs) / logits.shape[0]


def loss_and_gradients(x, targets, parameters):
    w1, b1, w2, b2 = parameters
    preactivation = x @ w1 + b1
    hidden = np.maximum(preactivation, 0.0)
    logits = hidden @ w2 + b2
    loss = cross_entropy(logits, targets)

    # Softmax + mean cross-entropy has a compact derivative.
    d_logits = (softmax(logits) - targets) / x.shape[0]
    d_w2 = hidden.T @ d_logits
    d_b2 = np.sum(d_logits, axis=0)
    d_hidden = (d_logits @ w2.T) * (preactivation > 0.0)
    d_w1 = x.T @ d_hidden
    d_b1 = np.sum(d_hidden, axis=0)
    return loss, (d_w1, d_b1, d_w2, d_b2)


def train_epoch(x, targets, parameters, rng, batch_size=64, learning_rate=0.1):
    order = rng.permutation(x.shape[0])
    total_loss = 0.0
    for start in range(0, x.shape[0], batch_size):
        batch = order[start:start + batch_size]
        loss, gradients = loss_and_gradients(x[batch], targets[batch], parameters)
        for parameter, gradient in zip(parameters, gradients, strict=True):
            parameter -= learning_rate * gradient
        total_loss += float(loss) * len(batch)
    return total_loss / x.shape[0]


def evaluate(x, targets, parameters):
    logits = forward(x, *parameters)
    accuracy = np.mean(np.argmax(logits, axis=1) == np.argmax(targets, axis=1))
    return float(cross_entropy(logits, targets)), float(accuracy)


def load_mnist(path, limit):
    """Read a Keras-format MNIST archive; Keras itself is not required."""
    with np.load(path, allow_pickle=False) as data:
        train_images, train_labels = data["x_train"][:limit], data["y_train"][:limit]
        test_images, test_labels = data["x_test"], data["y_test"]

    def prepare(images, labels):
        if images.ndim != 3 or images.shape[1:] != (28, 28) or images.dtype != np.uint8:
            raise ValueError("Expected uint8 MNIST images with shape [N, 28, 28]")
        if len(images) == 0 or labels.shape != (len(images),):
            raise ValueError("Each non-empty image split needs one label per image")
        x = images.reshape(len(images), 784).astype(np.float32) / np.float32(255.0)
        return x, one_hot(labels)

    return (*prepare(train_images, train_labels), *prepare(test_images, test_labels))


def synthetic_data(rng, count=512):
    """Learnable noisy patterns with MNIST shapes; these are NOT digit images."""
    prototypes = rng.uniform(0.0, 1.0, size=(10, 784)).astype(np.float32)

    def split(size):
        labels = rng.integers(0, 10, size=size)
        noise = rng.normal(0.0, 0.1, size=(size, 784)).astype(np.float32)
        x = np.clip(prototypes[labels] + noise, 0.0, 1.0)
        return x, one_hot(labels)

    return (*split(count), *split(128))


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Must be a positive integer")
    return number


def positive_float(value):
    number = float(value)
    if not np.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("Must be a finite positive number")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="Local MNIST .npz archive")
    parser.add_argument("--epochs", type=positive_int, default=3)
    parser.add_argument("--batch-size", type=positive_int, default=64)
    parser.add_argument("--learning-rate", type=positive_float, default=0.1)
    parser.add_argument(
        "--limit", type=positive_int, default=2048, help="Maximum training examples"
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    try:
        if args.data:
            x_train, y_train, x_test, y_test = load_mnist(args.data, args.limit)
            print(f"MNIST from {args.data}")
        else:
            x_train, y_train, x_test, y_test = synthetic_data(rng, min(args.limit, 512))
            print("Synthetic smoke data, NOT MNIST digits; accuracy is not an MNIST result.")
    except (OSError, KeyError, ValueError) as error:
        parser.exit(1, f"Cannot load data: {error}\n")

    parameters = initialize(rng)
    print(f"784 -> 32 -> 10 | train={len(x_train)} test={len(x_test)} | float32")
    loss, accuracy = evaluate(x_test, y_test, parameters)
    print(f"Initial: test_loss={loss:.4f} test_accuracy={accuracy:.1%}")
    for epoch in range(args.epochs):
        train_loss = train_epoch(
            x_train, y_train, parameters, rng, args.batch_size, args.learning_rate
        )
        loss, accuracy = evaluate(x_test, y_test, parameters)
        print(
            f"Epoch {epoch + 1}: train_loss={train_loss:.4f} "
            f"test_loss={loss:.4f} test_accuracy={accuracy:.1%}"
        )


if __name__ == "__main__":
    main()

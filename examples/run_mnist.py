"""Run the MNIST model using the NumPy CPU reference.

Use --forward-only for a single inference batch, or train with --epochs.
Without --data, inputs are synthetic patterns, not handwritten digits.
"""

import argparse
from pathlib import Path

import numpy as np
from mnist_cpu import evaluate, initialize, train_epoch
from mnist_data import load_mnist, synthetic_data
from mnist_forward import forward


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


def main(argv=None, *, forward_fn=forward):
    """The runtime target is forward_fn(x, w1, b1, w2, b2) -> float32 logits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--forward-only", action="store_true",
        help="Run one inference batch without training; use --batch-size for its size",
    )
    parser.add_argument("--data", type=Path, help="Local MNIST .npz archive")
    parser.add_argument("--epochs", type=positive_int, default=3)
    parser.add_argument("--batch-size", type=positive_int, default=64)
    parser.add_argument("--learning-rate", type=positive_float, default=0.1)
    parser.add_argument(
        "--limit", type=positive_int, default=2048, help="Maximum training examples"
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
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
    if args.forward_only:
        batch = x_test[:args.batch_size]
        logits = forward_fn(batch, *parameters)
        expected_shape = (len(batch), 10)
        if logits.shape != expected_shape or logits.dtype != np.float32:
            raise ValueError(
                f"Expected float32 logits {expected_shape}, got {logits.dtype} {logits.shape}"
            )
        if not np.isfinite(logits).all():
            raise ValueError("Forward pass returned non-finite logits")
        print(f"Forward pass: {batch.shape} -> {logits.shape} | {logits.dtype}")
        print(f"First sample logits: {logits[0].tolist()}")
        print(f"Predicted classes: {logits.argmax(axis=1).tolist()}")
        return

    print(f"784 -> 32 -> 10 | train={len(x_train)} test={len(x_test)} | float32")
    loss, accuracy = evaluate(x_test, y_test, parameters, forward_fn=forward_fn)
    print(f"Initial: test_loss={loss:.4f} test_accuracy={accuracy:.1%}")
    for epoch in range(args.epochs):
        train_loss = train_epoch(
            x_train, y_train, parameters, rng, args.batch_size, args.learning_rate
        )
        loss, accuracy = evaluate(x_test, y_test, parameters, forward_fn=forward_fn)
        print(
            f"Epoch {epoch + 1}: train_loss={train_loss:.4f} "
            f"test_loss={loss:.4f} test_accuracy={accuracy:.1%}"
        )


if __name__ == "__main__":
    main()

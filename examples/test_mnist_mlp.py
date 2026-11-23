"""Numerical checks; run: python3 -m unittest discover -s examples -p test_mnist_mlp.py."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from mnist_cpu import initialize, train_epoch
from mnist_data import load_mnist, one_hot
from mnist_forward import forward
from mnist_mlp import (
    cross_entropy,
    loss_and_gradients,
    softmax,
)
from run_mnist import main


class MnistTests(unittest.TestCase):
    def test_forward_only_calls_runtime_once_without_training(self):
        calls = []

        def runtime(x, w1, b1, w2, b2):
            values = (x, w1, b1, w2, b2)
            calls.append(values)
            self.assertEqual(
                [value.shape for value in values],
                [(4, 784), (784, 32), (32,), (32, 10), (10,)],
            )
            self.assertTrue(all(value.dtype == np.float32 for value in values))
            return forward(*values)

        output = io.StringIO()
        with patch("run_mnist.train_epoch", side_effect=AssertionError("must not train")):
            with contextlib.redirect_stdout(output):
                main(
                    ["--forward-only", "--batch-size", "4", "--limit", "16"],
                    forward_fn=runtime,
                )
        self.assertEqual(len(calls), 1)
        self.assertIn("Forward pass: (4, 784) -> (4, 10) | float32", output.getvalue())
        self.assertNotIn("Epoch", output.getvalue())

    def test_one_hot_and_invalid_labels(self):
        labels = np.array([0, 3, 9])
        encoded = one_hot(labels)
        self.assertEqual(encoded.shape, (3, 10))
        self.assertEqual(encoded.dtype, np.float32)
        np.testing.assert_array_equal(encoded.argmax(axis=1), labels)
        np.testing.assert_array_equal(encoded.sum(axis=1), np.ones(3))
        for invalid in [np.array([-1]), np.array([10]), np.array([1.5])]:
            with self.assertRaises(ValueError):
                one_hot(invalid)

    def test_softmax_and_loss_are_stable(self):
        logits = np.array([[1000.0, -1000.0, 0.0], [-1000.0, 1000.0, 0.0]])
        targets = one_hot(np.array([1, 0]), classes=3)
        probabilities = softmax(logits)
        self.assertTrue(np.isfinite(probabilities).all())
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
        np.testing.assert_allclose(cross_entropy(logits, targets), 2000.0)
        np.testing.assert_allclose(softmax(logits + 5000.0), probabilities)

    def test_all_gradients_against_finite_differences(self):
        rng = np.random.default_rng(7)
        parameters = initialize(rng, inputs=4, hidden=3, classes=2, dtype=np.float64)
        x = rng.normal(size=(5, 4))
        targets = one_hot(np.array([0, 1, 1, 0, 1]), classes=2).astype(np.float64)
        loss, gradients = loss_and_gradients(x, targets, parameters)
        np.testing.assert_allclose(loss, cross_entropy(forward(x, *parameters), targets))
        epsilon = 1e-5
        for parameter, analytic in zip(parameters, gradients, strict=True):
            numerical = np.empty_like(parameter)
            for index in np.ndindex(parameter.shape):
                original = parameter[index]
                parameter[index] = original + epsilon
                plus = cross_entropy(forward(x, *parameters), targets)
                parameter[index] = original - epsilon
                minus = cross_entropy(forward(x, *parameters), targets)
                parameter[index] = original
                numerical[index] = (plus - minus) / (2.0 * epsilon)
            np.testing.assert_allclose(analytic, numerical, rtol=1e-4, atol=1e-6)

    def test_training_reduces_loss_with_partial_final_batch(self):
        rng = np.random.default_rng(2)
        x = rng.normal(size=(19, 4)).astype(np.float32)
        targets = one_hot((x[:, 0] > 0).astype(np.int64), classes=2)
        parameters = initialize(rng, inputs=4, hidden=8, classes=2)
        initial = cross_entropy(forward(x, *parameters), targets)
        for _ in range(40):
            train_epoch(x, targets, parameters, rng, batch_size=7, learning_rate=0.1)
        final = cross_entropy(forward(x, *parameters), targets)
        self.assertLess(final, initial * 0.5)
        self.assertTrue(all(p.dtype == np.float32 for p in parameters))

    def test_mnist_archive_preprocessing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mnist.npz"
            np.savez(
                path,
                x_train=np.full((3, 28, 28), 255, dtype=np.uint8),
                y_train=np.array([1, 2, 3], dtype=np.uint8),
                x_test=np.zeros((2, 28, 28), dtype=np.uint8),
                y_test=np.array([4, 5], dtype=np.uint8),
            )
            x_train, y_train, x_test, y_test = load_mnist(path, limit=2)
            self.assertEqual(x_train.shape, (2, 784))
            self.assertEqual(x_train.dtype, np.float32)
            np.testing.assert_array_equal(x_train, np.ones((2, 784)))
            np.testing.assert_array_equal(y_train.argmax(axis=1), [1, 2])
            np.testing.assert_array_equal(x_test, np.zeros((2, 784)))
            np.testing.assert_array_equal(y_test.argmax(axis=1), [4, 5])


if __name__ == "__main__":
    unittest.main()

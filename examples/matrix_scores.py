"""Compiler goal: score three samples with a tiny two-class linear model.

Shapes: x [3, 2], weights [2, 2], bias [3, 2] -> scores [3, 2].
All values are float32. The illustrative weights are hand-picked, not trained.
Bias has the full output shape so this exercise needs no broadcasting.
Run `make matrix-demo` for the NumPy reference and expected results.
"""

import numpy as np


def compute(x, weights, bias):
    scores = x @ weights
    return scores + bias


if __name__ == "__main__":
    x = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float32)
    weights = np.array([[2, -1], [-1, 2]], dtype=np.float32)
    bias = np.array([[1, 0], [1, 0], [1, 0]], dtype=np.float32)
    expected = np.array([[3, -1], [0, 2], [2, 1]], dtype=np.float32)

    scores = compute(x, weights, bias)
    np.testing.assert_allclose(scores, expected, rtol=1e-6, atol=1e-6)
    assert scores.dtype == np.float32
    # Classification stays outside the compiled function for this milestone.
    classes = scores.argmax(axis=1)
    np.testing.assert_array_equal(classes, [0, 1, 0])

    print("NumPy reference: three samples, two class scores each")
    print(scores)
    print(f"Predicted classes: {classes.tolist()}")
    print("Reference checks passed. Next goal: reproduce these scores with PyTile.")

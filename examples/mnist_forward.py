"""First compiler target for the MNIST MLP: a pure NumPy forward pass.

Float32 shapes: x [batch, 784], w1 [784, 32], b1 [32], w2 [32, 10], b2 [10].
The result contains logits [batch, 10], not probabilities.
"""

import numpy as np


def forward(x, w1, b1, w2, b2):
    hidden = np.maximum(x @ w1 + b1, 0.0)
    logits = hidden @ w2 + b2
    return logits

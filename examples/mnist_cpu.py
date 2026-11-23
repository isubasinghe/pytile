"""NumPy CPU reference: initialization, minibatch training, and evaluation.

These orchestration helpers stay outside the compiled model functions.
"""

import numpy as np
from mnist_forward import forward
from mnist_mlp import cross_entropy, loss_and_gradients


def initialize(rng, inputs=784, hidden=32, classes=10, dtype=np.float32):
    w1 = (rng.standard_normal((inputs, hidden)) * np.sqrt(2.0 / inputs)).astype(dtype)
    b1 = np.zeros(hidden, dtype=dtype)
    w2 = (rng.standard_normal((hidden, classes)) * np.sqrt(2.0 / hidden)).astype(dtype)
    b2 = np.zeros(classes, dtype=dtype)
    return w1, b1, w2, b2


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


def evaluate(x, targets, parameters, *, forward_fn=forward):
    logits = forward_fn(x, *parameters)
    accuracy = np.mean(np.argmax(logits, axis=1) == np.argmax(targets, axis=1))
    return float(cross_entropy(logits, targets)), float(accuracy)

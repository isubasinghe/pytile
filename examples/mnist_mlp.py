"""Numerical training targets for the 784 -> 32 -> 10 MNIST MLP.

Compile mnist_forward.forward first, then these loss and gradient functions.
The CPU runner, parameter initialization, and data loading live separately.
"""

import numpy as np


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

"""MNIST archive loading, preprocessing, and synthetic data for host runners."""

import numpy as np


def one_hot(labels, classes=10):
    labels = np.asarray(labels)
    if labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Labels must be a 1D integer array")
    if np.any(labels < 0) or np.any(labels >= classes):
        raise ValueError(f"Labels must be in [0, {classes})")
    return np.eye(classes, dtype=np.float32)[labels]


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

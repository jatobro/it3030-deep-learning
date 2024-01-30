import numpy as np


def relu(x):
    return np.maximum(0, x)


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)

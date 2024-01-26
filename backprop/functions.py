import numpy as np


def relu(x):
    return max(0, x)


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

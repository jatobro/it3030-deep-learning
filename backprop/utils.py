import numpy as np

# helper functions


def broadcast(x, n):
    return x.reshape(-1, 1).repeat(n, axis=1)


def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)


# loss functions and their derivatives / gradients


def mse(z, t):
    return np.mean(np.square(z - t))


def d_mse(z, t):
    return (2 / len(z)) * (z - t)


# activations functions and their derivatives / gradients


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def d_sigmoid(z):
    return z * (1 - z)


def relu(x):
    return np.maximum(0, x)


def d_relu(x):
    if x > 0:
        return 1
    return 0


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


def d_softmax(x):
    s = softmax(x)

    jac_m = np.diag(s)

    for i in range(len(jac_m)):
        for j in range(len(jac_m)):
            if i == j:
                jac_m[i][j] = s[i] * (1 - s[i])
            else:
                jac_m[i][j] = -1 * s[i] * s[j]

    return jac_m


def tanh(x):
    return np.tanh(x)


def d_tanh(x):
    return 1 - np.tanh(x) ** 2


def lin(x):
    return x


def d_lin(x):
    return 1

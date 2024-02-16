import numpy as np

# helper functions


def train_val_test_split(X, y, train_size=0.6, val_size=0.2):
    first_split = int(len(X) * train_size)
    second_split = first_split + int(len(X) * val_size)

    train_X = X[:first_split]
    val_X = X[first_split:second_split]
    test_X = X[second_split:]

    train_y = y[:first_split]
    val_y = y[first_split:second_split]
    test_y = y[second_split:]

    return train_X, val_X, test_X, train_y, val_y, test_y


def broadcast(x, n):
    return x.reshape(-1, 1).repeat(n, axis=1)


def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)


# loss functions and their derivatives / gradients


def cross_entropy(z, t):
    return -np.sum(t * np.log(z))


def d_cross_entropy(z, t):
    return -t / z


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

from matplotlib import pyplot as plt
import numpy as np
from doodler import gen_standard_cases
from utils import mse, train_val_test_split
from network import Network
from layer import HiddenLayer, SoftmaxLayer, FlattenLayer

from config import (
    REG,
    REG_C,
    ACTIVATION,
    LAYER_COUNT,
    NEURON_COUNT,
    LOSS,
    LOSS_DERIVATIVE,
    CASES,
    EPOCHS,
    WEIGHT_RANGE,
    LEARNING_RATE,
)


TRAIN_CASES = int(CASES * 0.6)
VAL_CASES = int(CASES * 0.2)
TEST_CASES = int(CASES * 0.2)


def main():
    (images, targets_vectors, labels, (rows, cols), flat) = gen_standard_cases(
        count=CASES, show=False
    )

    (train_X, val_X, test_X, train_y, val_y, test_y) = train_val_test_split(
        images, targets_vectors, train_size=0.6, val_size=0.2
    )

    hidden_layers = [
        HiddenLayer(
            size=NEURON_COUNT,
            activation=ACTIVATION,
            weight_range=WEIGHT_RANGE,
            learning_rate=LEARNING_RATE,
        )
        for _ in range(LAYER_COUNT - 1)
    ]

    # init of network and all layers (softmax and hidden with weights and biases)
    network = Network(
        [FlattenLayer()]
        + hidden_layers
        + [
            HiddenLayer(
                size=train_y.shape[1], activation=ACTIVATION, weight_range=WEIGHT_RANGE
            ),
            SoftmaxLayer(),
        ]
    )

    # training the network

    train_loss = []  # loss from training
    val_loss = []

    for _ in range(EPOCHS):
        train_errors = []
        val_errors = []

        acc_weight_gradients = None
        acc_bias_gradients = None

        for i in range(TRAIN_CASES):
            # forward pass through network and get prediction with features for each case
            pred = network.forward_pass(features=train_X[i])

            # regularization
            reg = (
                sum(np.sum(np.abs(w)) for w in network.get_weights())
                if REG == "l1"
                else (
                    sum(np.sum(w**2) / 2 for w in network.get_weights())
                    if REG == "l2"
                    else 0
                )
            )

            # adding error for each case
            train_errors.append(LOSS(pred, train_y[i]) + reg * REG_C)

            # backward pass through network and get gradients
            weight_gradients, bias_gradients = network.backward_pass(
                LOSS_DERIVATIVE(pred, train_y[i])
            )

            # accumulate gradients to calculate mean between all cases

            if i == 0:
                acc_weight_gradients = [np.zeros_like(w) for w in weight_gradients]
                acc_bias_gradients = [np.zeros_like(b) for b in bias_gradients]

            acc_weight_gradients = [
                g + w for g, w in zip(acc_weight_gradients, weight_gradients)
            ]
            acc_bias_gradients += [
                g + b for g, b in zip(acc_bias_gradients, bias_gradients)
            ]

        # calculating mean gradients

        aggregated_weight_gradients = [g / CASES for g in acc_weight_gradients]
        aggregated_bias_gradients = [g / CASES for g in acc_bias_gradients]

        # tuning the network with mean gradients
        network.tune(aggregated_weight_gradients, aggregated_bias_gradients)

        # validation
        for i in range(VAL_CASES):
            pred = network.forward_pass(features=val_X[i])
            val_errors.append(mse(pred, val_y[i]))

        train_error = np.mean(train_errors)
        val_error = np.mean(val_errors)

        print(train_error)

        train_loss.append(train_error)
        val_loss.append(val_error)

    # testing the network

    test_errors_per_case = []
    for i in range(TEST_CASES):
        pred = network.forward_pass(features=test_X[i])
        test_errors_per_case.append(LOSS(pred, test_y[i]))

    test_loss = [np.mean(test_errors_per_case) for _ in range(EPOCHS)]

    print(test_loss[-1])

    # plotting the graph

    epochs = range(1, EPOCHS + 1)
    plt.plot(
        epochs,
        train_loss,
        color="blue",
        label="Train",
    )
    plt.plot(
        epochs,
        val_loss,
        color="yellow",
        label="Validation",
    )
    plt.plot(
        epochs,
        test_loss,
        color="red",
        label="Test",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()

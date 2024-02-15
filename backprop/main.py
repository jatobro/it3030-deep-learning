from matplotlib import pyplot as plt
import numpy as np
from utils import d_mse, mse
from network import Network
from layer import HiddenLayer, SoftmaxLayer

from config import REG, REG_C


EPOCHS = 1000
CASES = 1000


def main():
    features = np.random.randn(CASES, 5) * 10
    targets = np.random.randn(CASES, 3) * 10

    network = Network(
        [
            HiddenLayer(size=10),
            HiddenLayer(size=targets.shape[1]),
            SoftmaxLayer(),
        ]
    )

    error_per_epoch = []  # means of errors for all cases for each epoch

    for e in range(EPOCHS):
        errors = []  # list to store errors for each case

        acc_weight_gradients = None
        acc_bias_gradients = None

        for i in range(CASES):
            # forward pass through network and get prediction
            pred = network.forward_pass(features=features[i])

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
            errors.append(mse(pred, targets[i]) + reg * REG_C)

            # backward pass through network and get gradients
            weight_gradients, bias_gradients = network.backward_pass(
                d_mse(pred, targets[i])
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

        error = np.mean(errors)

        print(f"epoch {e + 1} error: {error}")

        # calculating mean gradients

        aggregated_weight_gradients = [g / CASES for g in acc_weight_gradients]
        aggregated_bias_gradients = [g / CASES for g in acc_bias_gradients]

        # tuning the network with mean gradients
        network.tune(aggregated_weight_gradients, aggregated_bias_gradients)

        error_per_epoch.append(error)

    # plotting the graph

    plt.plot(
        list(range(1, EPOCHS + 1)),
        error_per_epoch,
        marker="o",
        linestyle="-",
        color="b",
        label="MSE per Epoch",
    )
    plt.title("MSE vs. Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Mean MSE per case")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()

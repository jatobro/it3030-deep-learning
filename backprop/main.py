from matplotlib import pyplot as plt
import numpy as np
from utils import d_mse, mse, l1_reg
from network import Network
from layer import HiddenLayer, SoftmaxLayer
import yaml


EPOCHS = 100
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

    error_per_epoch = []

    for e in range(EPOCHS):
        errors = []

        acc_weight_gradients = None
        acc_bias_gradients = None

        for i in range(CASES):
            # forward pass through network and get prediction
            pred = network.forward_pass(features=features[i])

            errors.append(mse(pred, targets[i]))

            # backward pass through network and get gradients
            weight_gradients, bias_gradients = network.backward_pass(
                d_mse(pred, targets[i])
            )

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

        aggregated_weight_gradients = [g / CASES for g in acc_weight_gradients]
        aggregated_bias_gradients = [g / CASES for g in acc_bias_gradients]

        network.tune(aggregated_weight_gradients, aggregated_bias_gradients)

        error_per_epoch.append(error)

    epochs = list(range(1, EPOCHS + 1))

    # Plotting the graph

    plt.plot(
        epochs,
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

import numpy as np
from utils import d_mse, mse
from network import Network
from layer import HiddenLayer, SoftmaxLayer
import yaml


EPOCHS = 10000
CASES = 1000


def main():

    config = yaml.safe_load(open("./config.yml"))

    features = np.random.randn(CASES, 5) * 10
    targets = np.random.randn(CASES, 3) * 10

    network = Network(
        [
            HiddenLayer(size=10),
            HiddenLayer(size=targets.shape[1]),
            SoftmaxLayer(),
        ]
    )

    for _ in range(EPOCHS):
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

            # print(f"case {i + 1} weight gradients: {weight_gradients}")
            # print(f"case {i + 1} bias gradients: {bias_gradients}")
            # print()

        print(f"epoch {_ + 1} error: {np.mean(errors)}")

        aggregated_weight_gradients = [g / CASES for g in acc_weight_gradients]
        aggregated_bias_gradients = [g / CASES for g in acc_bias_gradients]

        network.tune(aggregated_weight_gradients, aggregated_bias_gradients)


if __name__ == "__main__":
    main()

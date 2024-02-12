import numpy as np
from utils import d_mse
from network import Network
from layer import HiddenLayer, SoftmaxLayer
import yaml

EPOCHS = 1
CASES = 7


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
        for i in range(CASES):
            pred = network.forward_pass(features=features[i])

            # print("pred:", pred)
            # print("mse:", mse(pred, targets[i]))

            gradients = network.backward_pass(d_mse(pred, targets[i]))
            print(gradients)


if __name__ == "__main__":
    main()

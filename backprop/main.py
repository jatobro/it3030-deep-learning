import numpy as np
from network import Network
from layer import InputLayer, HiddenLayer, OutputLayer
import yaml
from functions import mse


def main():
    config = yaml.safe_load(open("./config.yml"))

    features = np.random.randn(7, 5) * 10
    targets = np.random.randn(7, 1) * 10

    network = Network(
        [
            InputLayer(size=features.shape[1]),
            HiddenLayer(size=6),
            OutputLayer(activation="softmax"),
        ]
    )

    # training
    preds = network.forward_pass(features=features)
    error = mse(preds - targets)
    # network.backward_pass()

    print(error)


if __name__ == "__main__":
    main()

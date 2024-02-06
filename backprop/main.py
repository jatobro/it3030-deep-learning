import numpy as np
from network import Network
from layer import InputLayer, HiddenLayer, SoftmaxLayer
import yaml
from utils import mse


def main():
    config = yaml.safe_load(open("./config.yml"))

    features = np.random.randn(5) * 10
    target = np.random.randn(1) * 10

    network = Network(
        [
            InputLayer(size=len(features)),
            HiddenLayer(size=6),
            SoftmaxLayer(),
        ]
    )

    # training
    pred = network.forward_pass(features=features)
    error = mse(pred - target)
    # network.backward_pass()

    print("output:", pred, "\nerror:", error)


if __name__ == "__main__":
    main()

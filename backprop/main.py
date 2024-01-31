import numpy as np
from network import Network
from layer import InputLayer, HiddenLayer, OutputLayer
import yaml


def main():
    config = yaml.safe_load(open("./config.yml"))

    data = np.random.randn(7, 5) * 10

    network = Network(
        [
            InputLayer(size=data.shape[1]),
            HiddenLayer(size=6),
            OutputLayer(activation="softmax"),
        ]
    )

    # training
    pred = network.forward_pass(data=data)
    # network.backward_pass()
    print(pred)


if __name__ == "__main__":
    main()

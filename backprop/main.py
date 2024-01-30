from network import Network
from layer import InputLayer, HiddenLayer, OutputLayer
import yaml


def main():
    config = yaml.safe_load(open("./config.yml"))

    network = Network(
        [
            InputLayer(size=3),
            HiddenLayer(size=2),
            OutputLayer(activation="softmax"),
        ]
    )

    # training
    pred = network.forward_pass()
    # network.backward_pass()

    print(pred)


if __name__ == "__main__":
    main()

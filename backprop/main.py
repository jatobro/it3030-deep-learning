from network import Network
from layer import InputLayer, HiddenLayer, OutputLayer
from configparser import ConfigParser


def main():
    parser = ConfigParser()
    parser.read("config.ini")

    network = Network(
        [
            InputLayer(
                size=parser.getint("input", "size"),
                weight_range=parser.getfloat("input", "wr"),
                learning_rate=parser.getfloat("input", "lrate"),
            ),
            HiddenLayer(
                size=parser.getint("hidden_1", "size"),
                weight_range=parser.getfloat("hidden_1", "wr"),
                learning_rate=parser.getfloat("hidden_1", "lrate"),
            ),
            HiddenLayer(
                size=parser.getint("hidden_2", "size"),
            ),
            OutputLayer(size=parser.getint("output", "size")),
        ]
    )

    # training
    pred = network.forward_pass()
    # network.backward_pass()

    print(pred)


if __name__ == "__main__":
    main()

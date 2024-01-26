from network import Network
from layer import InputLayer, HiddenLayer, OutputLayer


def main():
    network = Network(
        [
            InputLayer(),
            HiddenLayer(),
            OutputLayer(),
        ]
    )

    # training
    pred = network.forward_pass()
    # network.backward_pass()

    print(pred)


if __name__ == "__main__":
    main()

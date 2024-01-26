import numpy as np
from functions import relu, softmax


class Layer:
    def __init__(
        self,
        input_size=5,
        size=5,
        activation=relu,
        is_output=False,
        weight_deviation=0.01,
    ):

        self.weights = np.random.randn(input_size, size) * weight_deviation
        self.biases = np.zeros(size)
        self.activation = activation
        self.is_output = is_output

    def forward_pass(self, inputs):
        outputs = []
        for input in inputs:
            outputs.append(
                list(
                    map(
                        self.activation,
                        np.dot(self.weights.T, input) + self.biases,
                    )
                )
            )

        return (
            np.array(list(map(softmax, outputs)))
            if self.is_output
            else np.array(outputs)
        )

    def backward_pass(self):
        pass

import numpy as np
from functions import relu


class Layer:
    def __init__(
        self, layer_sizes=(5, 5), activation_function=relu, weight_deviation=0.01
    ):

        self.weights = (
            np.random.randn(layer_sizes[0], layer_sizes[1]) * weight_deviation
        )
        self.biases = np.zeros(layer_sizes[0])
        self.activation_function = activation_function

    def forward_pass(self, inputs):
        outputs = []
        for input in inputs:
            outputs.append(
                list(
                    map(
                        self.activation_function,
                        np.dot(self.weights.T, input) + self.biases,
                    )
                )
            )

        return np.array(outputs)

    def backward_pass(self):
        pass

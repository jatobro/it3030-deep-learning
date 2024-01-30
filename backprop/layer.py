import numpy as np
from numpy._typing import NDArray
from functions import relu, softmax

from abc import ABC, abstractmethod


class Layer(ABC):
    def __init__(self, size=5, activation=relu, weight_range=0.05, learning_rate=0.01):
        self.size = size
        self.activation = activation
        self.weight_range = weight_range
        self.learning_rate = learning_rate

        self.weights = None
        self.biases = None

    @abstractmethod
    def forward_pass(self, inputs) -> NDArray[np.float64]:
        pass

    @abstractmethod
    def backward_pass(self):
        pass


class HiddenLayer(Layer):
    def forward_pass(self, inputs):
        """for hidden layers we create weights and biases (if they dont already exist) and calculate the output from these"""
        if self.weights is None:
            self.weights = (
                np.random.randn(inputs.shape[1], self.size) * self.weight_range
            )
            self.biases = np.zeros(self.size)

        return np.array(
            [
                self.activation(np.dot(self.weights.T, case) + self.biases)
                for case in inputs
            ]
        )

    def backward_pass(self):
        pass


class InputLayer(Layer):
    def forward_pass(self, inputs):
        """for input layers we just return the inputs"""
        return inputs

    def backward_pass(self):
        pass


class OutputLayer(Layer):
    def forward_pass(self, inputs):
        """for output layers we only apply softmax (or other activation function) to the inputs"""
        outputs = np.zeros_like(inputs)

        for i, case in enumerate(inputs):
            outputs[i] = softmax(case)

        return outputs

    def backward_pass(self):
        pass

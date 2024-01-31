import numpy as np
from numpy._typing import NDArray
from functions import relu, softmax, broadcast

from abc import ABC, abstractmethod


class Layer(ABC):
    def __init__(self, size, activation, learning_rate):
        self.size = size
        self.activation = (
            relu
            if activation == "relu"
            else softmax
            if activation == "softmax"
            else activation
        )

        self.learning_rate = learning_rate

    @abstractmethod
    def forward_pass(self, inputs) -> NDArray[np.float64]:
        pass

    @abstractmethod
    def backward_pass(self):
        pass


class HiddenLayer(Layer):
    def __init__(self, size=5, activation=relu, learning_rate=0.01, weight_range=0.05):
        super().__init__(size, activation, learning_rate)

        self.weight_range = weight_range

        self.weights = None
        self.biases = np.zeros(self.size)

    def init_weights(self, input_size):
        self.weights = np.random.randn(input_size, self.size) * self.weight_range

    def forward_pass(self, inputs):
        """for hidden layers we create weights (if they dont already exist) and calculate the output"""
        batch_size, input_size = inputs.shape

        if self.weights is None:
            self.init_weights(input_size)

        return self.activation(
            np.einsum("ij,jk->ik", inputs, self.weights) + self.biases
        )

    def backward_pass(self):
        pass


class InputLayer(Layer):
    def __init__(self, size=3, activation=relu, learning_rate=0.01):
        super().__init__(size, activation, learning_rate)

    def forward_pass(self, inputs):
        """for input layers we just return the inputs"""
        return inputs

    def backward_pass(self):
        pass


class OutputLayer(Layer):
    def __init__(self, size=3, activation=softmax, learning_rate=0.01):
        super().__init__(size, activation, learning_rate)

    def forward_pass(self, inputs):
        """for output layers we only apply softmax (or other activation function) to each case (each row) of the inputs"""
        return np.apply_along_axis(self.activation, axis=1, arr=inputs)

    def backward_pass(self):
        pass

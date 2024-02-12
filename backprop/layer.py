import numpy as np
from numpy._typing import NDArray
from utils import relu, softmax, d_sigmoid, d_relu

from abc import ABC, abstractmethod


class Layer(ABC):
    def __init__(self, size=3):
        self.size = size

    @abstractmethod
    def forward_pass(self, input) -> NDArray[np.float64]:
        pass

    @abstractmethod
    def backward_pass(self, input) -> NDArray[np.float64]:
        pass


class HiddenLayer(Layer):
    def __init__(self, size=5, activation=relu, learning_rate=0.01, weight_range=0.05):
        super().__init__(size)

        self.activation = (
            relu
            if activation == "relu"
            else softmax if activation == "softmax" else activation
        )
        self.d_activation = (
            d_relu
            if activation == "relu"
            else d_sigmoid if activation == "sigmoid" else None
        )
        self.learning_rate = learning_rate
        self.weight_range = weight_range

        self.weights = None
        self.biases = np.zeros(self.size)

    def forward_pass(self, input):
        """for hidden layers we create weights (if they dont already exist) and calculate the output"""
        if self.weights is None:
            self.weights = np.random.randn(len(input), self.size) * self.weight_range

        return self.activation(np.dot(input, self.weights) + self.biases)

    def backward_pass(self, input):
        """calculates the jacobi matrix (this layer respect to earlier layer, if first layer, respect to weights), dot this with the inputs and return the result"""
        j_z_sum = np.diag(self.d_activation(input))


class InputLayer(Layer):
    def forward_pass(self, input):
        """for input layers we just return the inputs"""
        return input

    def backward_pass(self, inputs):
        """for input layers we dont do anything in the backward pass because there are no weights fed into this layer"""
        pass


class SoftmaxLayer(Layer):
    def __init__(self, size=None):
        super().__init__(size)

        self.softmaxed_outputs = None

    def forward_pass(self, input):
        """for output layers we only apply softmax (or other activation function) to each case (each row) of the inputs"""
        self.size = len(input)

        self.softmaxed_outputs = softmax(input)
        return self.softmaxed_outputs

    def backward_pass(self, input):
        """calculates the softmax jacobi matrix, dot this with the inputs and return the result"""
        j_softmax_output = np.eye(self.size) * self.softmaxed_outputs + -1 * np.outer(
            self.softmaxed_outputs, self.softmaxed_outputs
        )  # TODO: check if this is correct

        return np.dot(input, j_softmax_output)

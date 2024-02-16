import numpy as np
from numpy._typing import NDArray
from utils import relu, softmax, sigmoid, d_sigmoid, d_relu
from config import REG_C, REG

from abc import ABC, abstractmethod


class Layer(ABC):
    def __init__(self, size):
        self.size = size

        self.inputs = None
        self.outputs = None

    @abstractmethod
    def forward_pass(self, inputs) -> NDArray[np.float64]:
        pass

    @abstractmethod
    def backward_pass(self, input_jacobian) -> NDArray[np.float64]:
        pass


class HiddenLayer(Layer):
    def __init__(
        self,
        size=5,
        activation=sigmoid,
        learning_rate=0.01,
        weight_range=0.05,
        d_activation=d_sigmoid,
    ):
        super().__init__(size)

        self.activation = (
            relu
            if activation == "relu"
            else softmax if activation == "softmax" else activation
        )
        self.d_activation = d_activation

        self.learning_rate = learning_rate
        self.weight_range = weight_range

        self.weights = None
        self.biases = np.zeros(self.size)

    def forward_pass(self, inputs):
        """for hidden layers we create weights (if they dont already exist) and calculate the output"""
        self.inputs = inputs

        if self.weights is None:
            self.weights = np.random.randn(len(inputs), self.size) * self.weight_range

        self.outputs = self.activation(np.dot(inputs, self.weights) + self.biases)

        return self.outputs

    def backward_pass(self, j_loss_outputs):
        """calculates the jacobi matrix (this layer respect to earlier layer, if first layer, respect to weights), dot this with the inputs and return the result"""
        j_outputs_weights = np.outer(self.inputs, self.d_activation(self.outputs))

        j_outputs_biases = self.d_activation(self.outputs)

        gradients = (
            (
                j_outputs_weights * j_loss_outputs + np.sign(self.weights)
                if REG == "l1"
                else self.weights if REG == "l2" else 0 * REG_C
            ),
            np.dot(j_outputs_biases, j_loss_outputs),
        )

        j_outputs_inputs = np.dot(self.d_activation(self.outputs), self.weights.T)

        return gradients, j_outputs_inputs

    def tune(self, weight_gradient, bias_gradient):
        """tunes the weights and biases of the layer"""
        self.weights -= self.learning_rate * weight_gradient
        self.biases -= self.learning_rate * bias_gradient

    def get_weights(self):
        return self.weights


class FlattenLayer(Layer):
    def __init__(self, size=None):
        super().__init__(size)

    def forward_pass(self, inputs):
        self.inputs = inputs
        self.outputs = inputs.flatten()
        return self.outputs

    def backward_pass(self, input_jacobian):
        pass


class SoftmaxLayer(Layer):
    def __init__(self, size=None):
        super().__init__(size)

    def forward_pass(self, inputs):
        """for output layers we only apply softmax (or other activation function) to each case (each row) of the inputs"""
        self.size = len(inputs)

        self.outputs = softmax(inputs)
        return self.outputs

    def backward_pass(self, j_loss_inputs):
        """calculates the softmax jacobi matrix, dot this with the inputs and return the result"""
        j_softmax_inputs = np.eye(self.size) * self.outputs + -1 * np.outer(
            self.outputs, self.outputs
        )  # TODO: check if this is correct

        return np.dot(j_loss_inputs, j_softmax_inputs)

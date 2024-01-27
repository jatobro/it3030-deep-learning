import numpy as np
from numpy._typing import NDArray
from functions import relu, softmax

from abc import ABC, abstractmethod


class Layer(ABC):
    def __init__(self, size=5, activation=relu):
        self.size = size
        self.activation = activation

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
        '''for hidden layers we create weights and biases and calculate the output from these'''
        if self.weights is None:
            self.weights = np.random.randn(inputs.shape[1], self.size) * 0.01
            self.biases = np.zeros(self.size)

        outputs =[]
        for case in inputs:
            outputs.append(
                list(
                    map(
                        self.activation,
                        np.dot(self.weights.T, case) + self.biases,
                    )
                )
            )

        return np.array(outputs)

    def backward_pass(self):
        pass


class InputLayer(Layer):
    def forward_pass(self, inputs):
        '''for input layers we just return the inputs'''
        return inputs

    def backward_pass(self):
        pass


class OutputLayer(Layer):
    def forward_pass(self, inputs):
        '''for output layers we only apply softmax (or other activation function) to the inputs'''
        outputs = np.zeros_like(inputs)

        for i, case in enumerate(inputs):
            outputs[i] = softmax(case)

        return outputs

    def backward_pass(self):
        pass

from layer import Layer
import numpy as np
from functions import softmax


class Network:
    def __init__(self, layers):
        self.layers = layers
        self.outputs = []

    def forward_pass(self, data=np.random.randn(3, 5) * 10):
        """forward pass of network calls forward pass of each layer and returns final output"""
        input = data
        for layer in self.layers:
            input = layer.forward_pass(input)
            self.outputs.append(input)

        return input

    def backward_pass(self):
        pass

from layer import Layer
import numpy as np
from functions import softmax


class Network:
    def __init__(self, layers):
        self.layers = layers

    def forward_pass(self, data=np.random.randn(3, 5) * 10):
        '''forward pass of network calls forward pass of each layer and returns final output'''
        output = data
        for layer in self.layers:
            output = layer.forward_pass(output)

        return output

    def backward_pass(self):
        pass

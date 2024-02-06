import numpy as np


class Network:
    def __init__(self, layers):
        self.layers = layers

    def forward_pass(self, features):
        """forward pass of network calls forward pass of each layer and returns final output"""
        outputs = features
        for layer in self.layers:
            outputs = layer.forward_pass(outputs)

        return outputs

    def backward_pass(self, jacobi_loss_output):
        """backward pass of network calls backward pass of each layer and returns final output"""
        outputs = jacobi_loss_output
        for layer in reversed(self.layers):
            outputs = layer.backward_pass(outputs)

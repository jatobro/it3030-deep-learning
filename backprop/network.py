import numpy as np
from layer import Layer


class Network:
    def __init__(self, layers: list[Layer]):
        self.layers = layers  # hidden layers and output layer

    def forward_pass(self, features):
        """forward pass of network calls forward pass of each layer and returns final output"""
        outputs = features
        for layer in self.layers:
            outputs = layer.forward_pass(outputs)

        return outputs

    def backward_pass(self, jacobi_loss_softmax):
        """backward pass of network calls backward pass of each layer and returns final output"""
        weight_gradients = []
        bias_gradients = []

        j_loss_output = self.layers[-1].backward_pass(
            jacobi_loss_softmax
        )  # backward through softmax layer (no weights)

        for layer in reversed(self.layers[:-1]):
            gradient, j_loss_output = layer.backward_pass(j_loss_output)

            weight_gradients.append(gradient[0])
            bias_gradients.append(gradient[1])

        return list(reversed(weight_gradients)), list(reversed(bias_gradients))

    def tune(self, weight_gradients, bias_gradients):
        """tunes the weights and biases of each layer"""
        for layer, w_gradient, b_gradient in zip(
            self.layers[:-1], weight_gradients, bias_gradients
        ):
            layer.tune(w_gradient, b_gradient)

    def get_weights(self):
        return [layer.get_weights() for layer in self.layers[:-1]]

from layer import Layer


class Network:
    def __init__(self, layers: list[Layer]):
        self.layers = layers

        self.outputs = []
        self.gradients = []

    def forward_pass(self, features):
        """forward pass of network calls forward pass of each layer and returns final output"""
        self.outputs = [features]
        for layer in self.layers[1:]:
            self.outputs.append(layer.forward_pass(self.outputs[-1]))

        return self.outputs[-1]

    def backward_pass(self, jacobi_loss_output):
        """backward pass of network calls backward pass of each layer and returns final output"""
        """
        outputs = jacobi_loss_output
        for layer in reversed(self.layers):
            outputs = layer.backward_pass(outputs)
        """
        softmax_layer = self.layers[-1]
        softmax_layer.backward_pass(jacobi_loss_output)

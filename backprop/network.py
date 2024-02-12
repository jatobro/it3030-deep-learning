from layer import Layer


class Network:
    def __init__(self, layers: list[Layer]):
        self.layers = layers

        self.outputs = []
        self.gradients = []

    def forward_pass(self, features):
        """forward pass of network calls forward pass of each layer and returns final output"""
        self.outputs = [features]
        for layer in self.layers:
            self.outputs.append(layer.forward_pass(self.outputs[-1]))

        return self.outputs[-1]

    def backward_pass(self, jacobi_loss_softmax):
        """backward pass of network calls backward pass of each layer and returns final output"""
        j_loss_output = self.layers[-1].backward_pass(
            jacobi_loss_softmax
        )  # backward through softmax layer (no weights)

        for layer in reversed(self.layers[:-1]):
            gradient, j_loss_output = layer.backward_pass(j_loss_output)
            self.gradients.append(gradient)

        return self.gradients

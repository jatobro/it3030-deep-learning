class Network:
    def __init__(self, layers):
        self.layers = layers
        self.outputs = []

    def forward_pass(self, features):
        """forward pass of network calls forward pass of each layer and returns final output"""
        self.outputs = [self.layers[0].forward_pass(features)]

        for layer in self.layers[1:]:
            self.outputs.append(layer.forward_pass(self.outputs[-1]))

        return self.outputs[-1]

    def backward_pass(self):
        pass

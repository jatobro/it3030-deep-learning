class Network:
    def __init__(self, layers):
        self.layers = layers
        self.outputs = []

    def forward_pass(self, data):
        """forward pass of network calls forward pass of each layer and returns final output"""
        for layer in self.layers:
            data = layer.forward_pass(data)
            self.outputs.append(data)

        return self.outputs[-1]

    def backward_pass(self):
        pass

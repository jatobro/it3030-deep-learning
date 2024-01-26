from layer import Layer
import numpy as np
from functions import softmax


class Network:
    def forward_pass(self, inputs=np.random.randn(3, 5) * 10):
        hidden_layer = Layer()
        output_layer = Layer(is_output=True)

        model = [hidden_layer, output_layer]

        output = inputs
        for layer in model:
            output = layer.forward_pass(output)

        return output

    def backward_pass(self):
        pass

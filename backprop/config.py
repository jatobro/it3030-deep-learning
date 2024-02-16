from utils import relu, mse, d_mse, cross_entropy, d_cross_entropy, tanh, d_tanh

LAYER_COUNT = 2
NEURON_COUNT = 100

ACTIVATION = tanh
D_ACTIVATION = d_tanh

LOSS = cross_entropy
LOSS_DERIVATIVE = d_cross_entropy

CASES = 500
EPOCHS = 10


REG = "l2"
REG_C = 0.001

WEIGHT_RANGE = 0.05
LEARNING_RATE = 0.01

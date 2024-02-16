from utils import relu, mse, d_mse

LAYER_COUNT = 5
NEURON_COUNT = 4

ACTIVATION = relu

LOSS = mse
LOSS_DERIVATIVE = d_mse

CASES = 1000
EPOCHS = 100


REG = "l1"
REG_C = 0.001

WEIGHT_RANGE = 0.05
LEARNING_RATE = 0.01

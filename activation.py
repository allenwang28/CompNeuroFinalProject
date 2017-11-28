import numpy as np


def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def ReLU(x):
    return np.maximum(x, 0, x)

def ReLU_derivative(x):
    return np.greater(x, 0).astype(int)

def softmax(x):
    # from https://gist.github.com/stober/1946926
    e = np.exp(x - np.max(x))
    return e / e.sum()

def softmax_derivative(x):
    return softmax(x) * (1 - softmax(x))

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1.0 - np.tanh(x)**2

def linear(x):
    return x

def linear_derivative(x):
    return 1

activation_map = {
    'sigmoid':sigmoid,
    'relu':ReLU,
    'softmax':softmax,
    'tanh':tanh,
    'linear':linear,
}

dactivation_map = {
    'sigmoid':sigmoid_derivative,
    'relu':ReLU_derivative,
    'softmax':softmax_derivative,
    'tanh':tanh_derivative,
    'linear':linear_derivative,
}

activation_choices = activation_map.keys()

class Activation:
    def __init__(self, activation):
        """
        activation:
            Choose from 'sigmoid', 'ReLU', 'softmax', 'tanh'.
        """
        activation = activation.lower()

        self.activation = activation_map[activation]
        self.activation_derivative = dactivation_map[activation]

    def activate(self, input_vec):
        return self.activation(input_vec)

    def dactivate(self, input_vec):
        return self.activation_derivative(input_vec)

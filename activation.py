import numpy as np

activation_choices = [
    'sigmoid',
    'ReLU',
    'softmax',
    'tanh'
]

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

class Activation:
    def __init__(self, activation):
        """
        activation:
            Choose from 'sigmoid', 'ReLU', 'softmax', 'tanh'.
        """
        activation = activation.lower()

        if activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_derivative = sigmoid_derivative
        elif activation == 'relu':
            self.activation = ReLU
            self.activation_derivative = ReLU_derivative
        elif activation == 'softmax':
            self.activation = softmax
            self.activation_derivative = softmax_derivative
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_derivative = tanh_derivative
        else:
            raise ValueError

    def activate(self, input_vec):
        return self.activation(input_vec)

    def dactivate(self, input_vec):
        return self.activation_derivative(input_vec)

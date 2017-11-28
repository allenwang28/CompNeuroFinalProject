import numpy as np
from activation import Activation

class Recurrent_Layer:
    def __init__(self, n_inputs, n_nodes, activation):
        """"
        Inputs:
            n_inputs:
                Number of inputs into the layer.
               
            n_nodes:
                Number of nodes in the layer.

            activation:
                Choose from 'sigmoid', 'ReLU', 'softmax', 'tanh'.

        Outputs:
            None
        """
        self.activation = Activation(activation)
        self.W = np.random.normal(0, 1, (n_inputs, n_nodes))
        self.b = np.zeros(n_nodes).reshape(n_nodes, 1)
        self.n_inputs = n_inputs
        self.n_nodes = n_nodes

    def activate(self, inputs):
        """
        Inputs:
            inputs:
                Input to the layer. Expect size n_inputs.

        Outputs:
            activation:
                The activation of the layer.
        """
        assert inputs.shape[0] == self.n_inputs

        self.inputs = inputs

        # Note - this reshaping is because sometimes np.dot results in a 1D vector which 
        # isn't what we want
        self.linear_value = (np.dot(self.W, inputs)).reshape(self.b.shape) + self.b
        self.activation = self.activation.activate(self.linear_value)
        return self.activation

    def compute_gradient(self, g):
        """
        Inputs:
            g:
                Error at the top layer.

        Notes:
            This function assumes that a forward prop just occured.

        Outputs:
            g:
                The updated error from this layer.

        """
        g = g * self.activation_derivative(self.linear_value)

        self.b_gradient = g
        self.w_gradient = np.dot(g, self.inputs.T)
        g = np.dot(self.W.T, g)




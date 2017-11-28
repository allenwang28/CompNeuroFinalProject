import numpy as np

from activation import Activation

# https://github.com/dennybritz/rnn-tutorial-rnnlm/blob/master/RNNLM.ipynb
# for reference.

# TODO - epochs?
# TODO - maybe add a debug flag but it would probably always be on anyways.

class RNN:
    def __init__(self,
                 input_layer_size,
                 hidden_layer_size, state_layer_activation,
                 output_layer_size, output_layer_activation,
                 bptt_truncate=None,
                 kernel=None,
                 eta=0.001,
                 rand=None,
                 verbose=0):
        """
        Notes:
            U - weight matrix from input into hidden layer.
            W - weight matrix from hidden layer to hidden layer.
            V - weight matrix from hidden layer to output layer.

        Inputs:
            input_size:
                Size of the input vector. We expect a 2D numpy array, so this should be X.shape[1]

            hidden_layer_size:
                Hidden layer size.

            state_layer_activation:
                Activation of s_t. Choose from 'sigmoid', 'ReLU', 'softmax', 'tanh'.

            output_size:
                Size of the output vector. We expect a 2D numpy array, so this should be Y.shape[1]

            output_layer_activation:
                Activation of o_t. Choose from 'sigmoid', 'ReLU', 'softmax', 'tanh'.

            bptt_truncate(opt):
                If left at None, back propagation through time will be applied for all time steps. 

                Otherwise, a value for bptt_truncate means that 
                bptt will only be applied for at most bptt_truncate steps.

            kernel(opt):
                If left at None, this becomes a regular RNN. 
                # TODO - fill this

            eta (opt):
                Learning rate. Initialized to 0.001.

            rand (opt):
                Random seed. Initialized to None (no random seed).

            verbose (opt):
                Verbosity: levels 0 - 2

        Outputs:
            None
        """
        np.random.seed(rand)

        self.input_layer_size = input_layer_size

        self.hidden_layer_size = hidden_layer_size
        self.state_layer_activation = state_layer_activation
        self.state_activation = Activation(state_layer_activation)

        self.output_layer_size = output_layer_size
        self.output_layer_activation = output_layer_activation
        self.output_activation = Activation(output_layer_activation)

        self.kernel = kernel
        self.bptt_truncate = bptt_truncate

        # U - weight matrix from input into hidden layer.
        # W - weight matrix from hidden layer to hidden layer.
        # V - weight matrix from hidden layer to output layer.
        self.U = np.random.uniform(-np.sqrt(1./input_layer_size),
                                    np.sqrt(1./input_layer_size), 
                                    (hidden_layer_size, input_layer_size))
        self.V = np.random.uniform(-np.sqrt(1./hidden_layer_size),
                                    np.sqrt(1./hidden_layer_size),
                                    (output_layer_size, hidden_layer_size))
        self.W = np.random.uniform(-np.sqrt(1./hidden_layer_size),
                                    np.sqrt(1./hidden_layer_size),
                                    (hidden_layer_size, hidden_layer_size))

        self.eta = eta
        self.verbose = verbose

    def fit(self, X_train, y_train):
        """
        Inputs:
            X_train:
                Training inputs. Expect size (input_layer_size, N) where N is the number of samples.

            Y_train:
                Training outputs. Expect size (output_layer_size, N) where N is the number of samples.

        Outputs:
            None
        """
        pass
    
    def forward_propagation(self, x):
        """
        Outputs:
            o:
                The activation of the output layer.
            s:
                The activation of the hidden state. 
        """
        T = x.shape[0]

        s = np.zeros((T + 1, self.hidden_layer_size))
        o = np.zeros((T, self.output_layer_size))
        s_linear = np.zeros((T + 1, self.hidden_layer_size))
        o_linear = np.zeros((T, self.output_layer_size))
        
        for t in np.arange(T):
            state_linear = np.dot(self.U, x[t]) + np.dot(self.W, s[t-1])
            s_linear[t] = state_linear
            s[t] = self.state_layer_activation.activate(state_linear)
            output_linear = np.dot(self.V, s[t])
            o[t] = self.output_layer_activation.activate(output_linear)
            o_linear[t] = output_linear
        return (o, s, s_linear, o_linear)

    def back_propagation_through_time(self, x, y):
        T = len(y)
        assert T == len(x)
        
        if self.bptt_truncate is None:
            bptt_truncate = T
        else:
            bptt_truncate = self.bptt_truncate

        o, s, s_linear, o_linear = self.forward_propagation(x)

        dLdU = np.zeros(self.U.shape)
        dLdV = np.zeros(self.V.shape)
        dLdW = np.zeros(self.W.shape)

        delta_o = y - o
        for t in np.arange(T)[::-1]:
            # Backprop the error at the output layer
            g = delta_o[t] * self.output_activation.dactivate(o_linear[t])
            dLdV += np.dot(g, s[t].T)
            delta_t = np.dot(self.V.T, g)

            # Backpropagation through time for at most bptt truncate steps
            for bptt_step in np.arange(max(0, t - bptt_truncate), t+1)[::-1]:
                g = g  * self.state_activation.dactivate()
                dLdW += 




if __name__ == "__main__":
    x = np.array([[1,1], [2,2], [3,3], [4,4]])
    y = np.array([[1,1], [2,2], [3,3], [4,4]])
    input_size = x.shape[1]
    hidden_layer_size = 2
    output_size = y.shape[1]
    state_layer_activation = 'sigmoid'
    output_layer_activation = 'tanh'

    rnn = RNN(input_size, hidden_layer_size, state_layer_activation, 
              output_size, output_layer_activation)
    o, s = rnn.forward_propagation(x)

    print o

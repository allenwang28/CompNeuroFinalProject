import numpy as np

from rnn import RNN
import activation

import argparse
import pickle

import os
import matplotlib.pyplot as plt

import itertools
from multiprocessing import Pool
import copy_reg
import types

# Use this for multithreading in compete mode
def _pickle_method(method):
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = method.im_class
    return _unpickle_method, (func_name, obj, cls)

def _unpickle_method(func_name, obj, cls):
    for cls in cls.mro():
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj, cls)

copy_reg.pickle(types.MethodType, _pickle_method, _unpickle_method)

def fit_rnn(params):
    rnn = params['rnn']
    label = params['label']
    X = params['X']
    y = params['y']
    return_map = {}

    tr_loss, te_loss = rnn.fit(X, y)
    print "{0} finished training!".format(label)
    return_map['tr_loss'] = tr_loss
    return_map['te_loss'] = te_loss
    return_map['W'] = rnn.W
    return_map['label'] = label
    return return_map

def fit_rnn_star(params):
    return fit_rnn(*params)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('learning_rule', 
                        choices=['modified', 'bptt', 'fa', 'dfa'],                    
                        action='store',
                        help="Choose between \'bptt\' \'fa\' \'dfa\' and \'modified\'")

    parser.add_argument('--mode',
                        choices=['normal','compete'],
                        action='store',
                        default='normal',
                        help='Either choose a learning rule or compare the gradients for different learning rules')

    parser.add_argument('--training_data_path',
                        default='none',
                        type=str,
                        help='Path to pickled training data')

    # Optional args
    parser.add_argument('--bptt_truncate',
                        type=int,
                        action='store',
                        default=None,
                        help='Truncation for BPTT - Not providing this means there is no truncation')

    parser.add_argument('--kernel', help='TODO')
    parser.add_argument('--tau', type=float,
                        default=10000.)

    parser.add_argument('--validation_size', type=float,
                        default=0.3)


    parser.add_argument('--state_layer_activation',
                        choices=activation.activation_choices,
                        action='store',
                        help='State layer activation',
                        default='sigmoid')
    parser.add_argument('--output_layer_activation',
                        choices=activation.activation_choices,
                        action='store',
                        help='Output layer activation',
                        default='linear')
    parser.add_argument('--state_layer_size',
                        action='store',
                        type=int,
                        help='state layer size',
                        default=2)

    parser.add_argument('--eta', help='Learning Rate', type=float, default=0.001)
    parser.add_argument('--epochs', help='Epochs', type=int, default=1000)
    parser.add_argument('--v', '--verbose',
                        type=int,
                        help='Between 0-2', 
                        dest='verbose')

    parser.add_argument('--rand',
                        type=int,
                        help='Random seed',
                        default=None)

    args = parser.parse_args()

    # TODO - add parameter in for this later
    input_layer_size = 2
    output_layer_size = input_layer_size

    trajectories = None
    if os.path.exists(args.training_data_path):
        with open(args.training_data_path, 'rb') as f:
            trajectories = pickle.load(f)
            if not trajectories:
                raise Exception('Invalid pkl file')
            X = []
            Y = []
            for trajectory in trajectories:
                X.append(np.array(trajectory[:-1])/600 - 0.5)
                Y.append(np.array(trajectory[1:])/600 - 0.5)
    else:
        X = []
        Y = []
        for i in range(10):
            x = np.random.normal(0, 1, (5,2))
            y = x / 10.
            X.append(x)
            Y.append(y)

    if args.mode == 'normal':
        rnn = RNN(input_layer_size,
              args.state_layer_size, args.state_layer_activation,
              output_layer_size, args.output_layer_activation,
              epochs=args.epochs,
              bptt_truncate = args.bptt_truncate,
              learning_rule = args.learning_rule,
              tau = args.tau,
              eta=args.eta,
              rand=args.rand,
              verbose=args.verbose)
        training_losses, validation_losses = rnn.fit(X, Y, validation_size=args.validation_size)   
        plt.plot(range(args.epochs),training_losses, label='Training Loss')
        plt.plot(range(args.epochs),validation_losses, label='Validation Loss')
        plt.title("Training and validation losses for {0}".format(args.learning_rule))
        plt.legend()
        plt.show()
    elif args.mode == 'compete':
        labels = ['bptt', 'fa', 'dfa', 'modified']
        # Turn off verbosity because of multiprocessing...
        args.verbose = 0
        rnn_bptt = RNN(input_layer_size,
                   args.state_layer_size, args.state_layer_activation,
                   output_layer_size, args.output_layer_activation,
                   epochs=args.epochs,
                   bptt_truncate = args.bptt_truncate,
                   learning_rule = 'bptt',
                   tau = args.tau,
                   eta=args.eta,
                   rand=args.rand,
                   verbose=args.verbose)       
        rnn_fa  =  RNN(input_layer_size,
                   args.state_layer_size, args.state_layer_activation,
                   output_layer_size, args.output_layer_activation,
                   epochs=args.epochs,
                   bptt_truncate = args.bptt_truncate,
                   learning_rule = 'fa',
                   tau = args.tau,
                   eta=args.eta,
                   rand=args.rand,
                   verbose=args.verbose)       
        rnn_dfa = RNN(input_layer_size,
                   args.state_layer_size, args.state_layer_activation,
                   output_layer_size, args.output_layer_activation,
                   epochs=args.epochs,
                   bptt_truncate = args.bptt_truncate,
                   learning_rule = 'dfa',
                   tau = args.tau,
                   eta=args.eta,
                   rand=args.rand,
                   verbose=args.verbose)       
        rnn_m   =  RNN(input_layer_size,
                   args.state_layer_size, args.state_layer_activation,
                   output_layer_size, args.output_layer_activation,
                   epochs=args.epochs,
                   bptt_truncate = args.bptt_truncate,
                   learning_rule = 'modified',
                   tau = args.tau,
                   eta=args.eta,
                   rand=args.rand,
                   verbose=args.verbose)  
        # Initialize all RNNs to start with the same random values.
        rnns = [rnn_bptt, rnn_fa, rnn_dfa, rnn_m]
        W = rnn_fa.W
        U = rnn_fa.U
        V = rnn_fa.V
        B = rnn_fa.B
        for rnn in rnns:
            rnn.W = W.copy()
            rnn.U = U.copy()
            rnn.V = V.copy()
            rnn.B = B.copy()

        param_maps = []
        # Prepare data for the pool
        for rnn, label in zip(rnns, labels):
            param_map = {
                    'rnn': rnn,
                    'label': label,
                    'X': X,
                    'y': Y,
                    }
            param_maps.append(param_map)

        print "Beginning training"
        results = Pool(len(labels)).map(fit_rnn, param_maps)
        print "Fits completed!"
        fig, (ax1, ax2) = plt.subplots(2, 1)

        for result in results:
            W = result['W']
            tr_loss = result['tr_loss']
            te_loss = result['te_loss']
            label = result['label']
            print ("Weight matrix for {0}:\n {1}\n".format(label, W))
            ax1.plot(range(args.epochs), tr_loss, label=label)
            ax1.set_title('Training Losses')
            ax2.plot(range(args.epochs), te_loss, label=label)
            ax2.set_title('Test Losses')

        plt.legend()
        plt.show()



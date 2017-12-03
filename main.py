import numpy as np

from rnn import RNN
import activation

import argparse
import pickle

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('learning_rule', 
                        choices=['modified', 'bptt'],                    
                        action='store',
                        help="Choose between \'bptt\' and \'modified\'")

    parser.add_argument('--training_data_path',
                        type=str,
                        help='Path to pickled training data')

    # Optional args
    parser.add_argument('--bptt_truncate',
                        type=int,
                        action='store',
                        default=None,
                        help='Truncation for BPTT - Not providing this means there is no truncation')

    parser.add_argument('--kernel', help='TODO')


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

    # TODO - add arguments for simulations

    if args.learning_rule == 'bptt':
        kernel = None
    elif args.learning_rule == 'modified':
        # TODO - modify this later
        kernel = None

    input_layer_size = 2
    output_layer_size = input_layer_size

    rnn = RNN(input_layer_size,
              args.state_layer_size, args.state_layer_activation,
              output_layer_size, args.output_layer_activation,
              epochs=args.epochs,
              bptt_truncate = args.bptt_truncate,
              learning_rule = args.learning_rule,
              kernel = kernel,
              eta=args.eta,
              rand=args.rand,
              verbose=args.verbose)

    # TODO - dummy placeholder simulation below
    #with open(args.training_data_path, 'wb') as f:
    #  trajectories = pickle.load(f)
    
    X = []
    Y = []
    for i in range(10):
        x = np.random.normal(0, 1, (5,2))
        y = x / 10.

        X.append(x)
        Y.append(y)

    print "Before training:"
    print "MSE:"
    print rnn.score(X, Y)

    rnn.fit(X, Y)   
    print "After training:"
    print "MSE:"
    print rnn.score(X, Y)



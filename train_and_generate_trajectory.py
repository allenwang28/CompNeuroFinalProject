import argparse
import pickle

import numpy as np

from rnn import RNN


if __name__=='__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('train_data_path', type=str)
  parser.parse_args()

  rnn_modified = RNN(2, 6, 'tanh', 2, 'linear', 2, 100, None, 'modified', 5000, 1e-4, None, 2)
  with open(args.training_data_path, 'rb') as f:
    trajectories = pickle.load(f)
  X = [np.array(trajectory[:-1])/600 - 1/2 for trajectory in trajectories]
  Y = [np.array(trajectory[1:])/600 - 1/2 for trajectory in trajectories]
  rnn_modified.fit(X, Y)
  seed_x = [np.array([[0, 0]])]
  preds = [seed_x[0]]
  for i in range(len(trajectories[0])):
    preds.append(rnn_modified.predict(preds[-1])[0])
  with open('rnn_pred_traj.pkl', 'rb') as f:
    pickle.dump(f, preds, protocol=2)


import sys
sys.path.append('../')
import numpy as np
if sys.version_info[0] < 3:
  sys.path.append('../src/data_utils')
  import DataHandler as dh
else:
  from src.data_utils import DataHandler as dh
from parser import *
import argparse

def parse_args():
  parser = argparse.ArgumentParser(description='data recording')

  parser.add_argument('--data_path', help='Path to directory that comprises the training data (default=" ").',
                      type=str, default='../data/')
  parser.add_argument('--save_path', help='Path to directory that saves pickle data (default=" ").',
                      type=str,
                      default='../data/data.pickle')
  parser.add_argument('--batch_size', help='Batch size for training (default=32).', type=int, default=16)
  parser.add_argument('--min_buffer_size', help='Minimum buffer size (default=1000).', type=int, default=1000)
  parser.add_argument('--max_buffer_size', help='Maximum buffer size (default=100k).', type=int, default=100000)
  parser.add_argument('--truncated_backprop_length', help='truncated back propagation length', type=int, default=10)
  parser.add_argument('--output_sequence_length', help='output sequence length', type=int, default=1)
  parser.add_argument('--cmd_vector_dim', help='robot command vector dimension', type=int, default=2)
  parser.add_argument('--goal_x', help='x coordinate of robot goal state', type=float, default=23.00)
  parser.add_argument('--goal_y', help='y coordinate of robot goal state', type=float, default=0.00)
  parser.add_argument('--prediction_horizon', help='prediction horizon', type=int, default=15)
  parser.add_argument('--bag_file', help='csv file containg control commands (default=" ").',
                      type=str, default='data_2019-08-20-12-29-52.bag')
  parser.add_argument('--distance_threshold', help='Distance threshold to start new trajectory', type=float, default=1)
  parser.add_argument('--topics_config', help='yaml file containg subscription topics (default=" ").',
                      type=str, default='../config/topics.yaml')



  args = parser.parse_args()

  return args

args = parse_args()

data_prep = dh.DataHandler(args)

data_prep.process_simulation_data_()

for i in range(1000):
  batch_cmd, batch_pred_cmd, batch_robot_state, new_epoch = data_prep.getBatch()
  print("New epoch: " + str(new_epoch))
  if np.any(data_prep.sequence_reset):
    for sequence_idx in range(data_prep.sequence_reset.shape[0]):
      if data_prep.sequence_reset[sequence_idx] == 1:
        print("Reset state: " + str(sequence_idx))

print("successfully obtained training data batch")

print("#########")

data_prep.save_data()

print("pickle data saved")



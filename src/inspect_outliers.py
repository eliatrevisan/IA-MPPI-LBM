import sys
import os
from cv2 import log
sys.path.append('../')
import numpy as np
import argparse
import pylab as pl
import json
import importlib
if sys.version_info[0] < 3:
	print("Using Python " + str(sys.version_info[0]))
	sys.path.append('../src/data_utils')
	sys.path.append('../src/models')
	import DataHandler as dh
	import DataHandlerLSTM as dhlstm
	from plot_utils import *
	import Support as sup
	from utils import *
else:
	print("Using Python " + str(sys.version_info[0]))
	from src.data_utils import DataHandler as dh
	from src.data_utils import DataHandlerLSTM as dhlstm
	from src.data_utils import DataHandlerRoboat as dhroboat
	from src.data_utils.plot_utils import *
	from src.data_utils import Support as sup
	from src.data_utils.utils import *
from datetime import datetime
import pickle as pkl
import time
from copy import deepcopy
from multiprocessing.pool import ThreadPool
import colorama
from colorama import Fore, Style
import matplotlib.pyplot as plt

# Model directories
def parse_args():
	parser = argparse.ArgumentParser(description='LSTM model training')

	parser.add_argument('--model_name',
											help='Path to directory that comprises the model (default="model_name").',
											type=str, default="VGDNN")
	parser.add_argument('--num_test_sequences', help='Number of test sequences', type=int, default=10)
	parser.add_argument('--exp_num', help='Experiment number', type=int, default=247)
	parser.add_argument('--n_samples', help='Number of samples', type=int, default=10)
	parser.add_argument('--scenario', help='Scenario of the dataset (default="").',
											type=str, default="datasets/ewap_dataset/seq_eth")
	parser.add_argument('--record', help='Is grid rotated? (default=True).', type=sup.str2bool,
											default=True)
	parser.add_argument('--save_figs', help='Save figures?', type=sup.str2bool,
											default=True)
	parser.add_argument('--noise_cell_state', help='Adding noise to cell state of the agent', type=float,
											default=0)
	parser.add_argument('--noise_cell_grid', help='Adding noise to cell state of the grid', type=float,
											default=5)
	parser.add_argument('--real_world_data', help='real_world_data', type=sup.str2bool,
											default=False)
	parser.add_argument('--update_state', help='update_state', type=sup.str2bool,
											default=False)
	parser.add_argument('--gpu', help='Enable GPU training.', type=sup.str2bool,
											default=False)
	parser.add_argument('--freeze_other_agents', help='FReeze other agents.', type=sup.str2bool,
											default=False)
	parser.add_argument('--unit_testing', help='Run  Unit Tests.', type=sup.str2bool,
											default=False)
	parser.add_argument('--constant_velocity', help='Run CV comparison', type=sup.str2bool,
											default=False)
	args = parser.parse_args()

	return args


test_args = parse_args()

if test_args.gpu:
	import tensorflow as tf
else:
	os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
	import tensorflow as tf

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

cwd = os.getcwd()

model_path = os.path.normpath(cwd + '/../') + '/trained_models/' + test_args.model_name + "/" + str(test_args.exp_num)

print("Loading data from: '{}'".format(model_path))
file = open(model_path + '/model_parameters.pkl', 'rb')
if sys.version_info[0] < 3:
	model_parameters = pkl.load(file)  # ,encoding='latin1')
else:
	model_parameters = pkl.load(file, encoding='latin1')
file.close()
args = model_parameters["args"]

with open(args.model_path + '/model_parameters.json', 'w') as f:
	json.dump(args.__dict__,f)

print("Training steps: ", args.total_training_steps)

# change some args because we are doing inference
truncated_backprop_length = args.truncated_backprop_length
args.truncated_backprop_length = 8
args.batch_size = 16
args.keep_prob = 1.0

training_scenario = args.scenario
args.scenario = 'herengracht'
args.real_world_data = test_args.real_world_data
args.dataset = '/' + args.scenario + '.pkl'
# Specify the number of datasets / scenarios 
n_scenarios = 4

# Create Datahandler class for Herengracht experiment
original_scenario = args.scenario
data_prep = dhlstm.DataHandlerLSTM(args)
map_args = {"file_name": 'map.png',
				"resolution": 0.081,
				"map_size": np.array([1715,881]),
				"map_center": np.array([-78,-40])}
data_prep.processData(**map_args)

# Create Datahandler class for Prinsengracht scenario
args.scenario = "prinsengracht"
data_prep_2 = dhlstm.DataHandlerLSTM(args)
map_args = {"file_name": 'map.png',
				"resolution": 0.091,
				"map_size": np.array([816,816]),
				"map_center": np.array([-37.1, -37.1])} 
data_prep_2.processData(**map_args)

args.scenario = "bloemgracht"
data_prep_3 = dhlstm.DataHandlerLSTM(args)
map_args = {"file_name": 'map.png',
				"resolution": 0.22,
				"map_size": np.array([1469,928]),
				"map_center": np.array([-161.59,-102.08])} 
data_prep_3.processData(**map_args)

args.scenario = "open_crossing"
data_prep_4 = dhlstm.DataHandlerLSTM(args)
map_args = {"file_name": 'map.png',
				"resolution": 0.6,
				"map_size": np.array([100,100]),
				"map_center": np.array([-30,-30])} 
data_prep_4.processData(**map_args)

args.scenario = original_scenario

# Initialize main data handler
dpreps = [data_prep, data_prep_2, data_prep_3, data_prep_4]
data_roboat = dhroboat.DataHandlerRoboat(args)

# Import Deep Learning model
#module = importlib.import_module("src.models."+args.model_name)
#globals().update(module.__dict__)

# Create Model Graph
#model = NetworkModel(args)

#config = tf.ConfigProto()

def visualizeBatch(dpreps, data_log):
	dset = data_log[0]
	traj_idx = data_log[1]

	batch_size = 16
	for i in range(batch_size):

		fig, ax = plt.subplots()
		sup.plot_grid_roboat(ax, dpreps[dset[i]].agent_container.occupancy_grid.center, dpreps[dset[i]].agent_container.occupancy_grid.gridmap, dpreps[dset[i]].agent_container.occupancy_grid.resolution, dpreps[dset[i]].agent_container.occupancy_grid.map_size)
		_, traj = dpreps[dset[i]].trajectory_set[traj_idx[i]]
		otherv = traj.other_agents_positions
		ax.plot(traj.pose_vec[:,0], traj.pose_vec[:,1])

		for t in range(len(otherv)-1):
			for v in range(len(otherv[t])):
				ax.scatter(otherv[t][v][0], otherv[t][v][1])

		plt.show()
		print("Scenario: " + i + " / " + batch_size)
	
	print("done")

def plotBatch(dpreps, data_log):
	dset = data_log[1]
	agent_ids = data_log[0]

	batch_size = 16
	for i in range(batch_size):

		fig, ax = plt.subplots()
		sup.plot_grid_roboat(ax, dpreps[dset[i]].agent_container.occupancy_grid.center, dpreps[dset[i]].agent_container.occupancy_grid.gridmap, dpreps[dset[i]].agent_container.occupancy_grid.resolution, dpreps[dset[i]].agent_container.occupancy_grid.map_size)
		traj = dpreps[dset[i]].agent_container.getAgentTrajectories(agent_ids[i])[0] 
		otherv = traj.other_agents_positions
		ax.plot(traj.pose_vec[:,0], traj.pose_vec[:,1])

		for t in range(len(otherv)-1):
			for v in range(len(otherv[t])):
				ax.scatter(otherv[t][v][0], otherv[t][v][1])

		plt.show()
		print("Scenario: " + str(i) + " / " + str(batch_size))


#rainLog = [[1, 3, 3, 1, 1, 3, 0, 3, 0, 2, 2, 0, 3, 3, 2, 2], [209, 222, 212, 223, 210, 217, 205, 220, 201, 202, 218, 207, 206, 204, 221, 211]]
#ValidationLog = [[1, 1, 2, 3, 0, 3, 3, 1, 0, 1, 2, 1, 0, 2, 3, 3], [3558, 3513, 2532, 3549, 2550, 1955, 3477, 3567, 1962, 3576, 2556, 3585, 3531, 3540, 2993, 1969]]

#TrainLog = [[1410, 1132, 3391, 1571, 3260, 1369, 3511, 2476, 1769, 2167, 2772, 1312, 1615, 602, 946, 2301], 3]

# Outlier 1
#ValidationLog = [[1155, 858, 103, 1830, 2998, 243, 275, 1327, 2286, 1954, 110, 457, 552, 2466, 2421, 811], [1, 2, 0, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 1, 1, 2]]

# Outlier 2
ValidationLog = [[1968, 1037, 1138, 2527, 39, 244, 3737, 588, 3606, 356, 547, 1266, 3217, 2023, 3364, 2903], [0, 3, 3, 1, 3, 3, 1, 3, 1, 1, 1, 0, 2, 1, 1, 1]]


# Get Batch of Data
#train_dict = data_roboat.getPredefinedBatch(dpreps, TrainLog)
#validation_dict = data_roboat.getPredefinedBatch(dpreps, ValidationLog)

#visualizeBatch(dpreps, TrainLog)
#visualizeBatch(dpreps, ValidationLog)
plotBatch(dpreps, ValidationLog)





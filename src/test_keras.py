import sys
import os

sys.path.append('../')
import numpy as np
import argparse
import pylab as pl
import pickle as pkl
# import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import random
from copy import deepcopy

if sys.version_info[0] < 3:
	sys.path.append(sys.path[0] + '/data_utils') # sys.path.append('../src/data_utils')
	sys.path.append(sys.path[0] + '/models') # sys.path.append('../src/models')
	import DataHandler_Keras as dh
	import DataHandlerLSTM as dhlstm
	from plot_utils import *
	import Support as sup
	from Performance import *
	from utils import *
	import Recorder as rec
else:
	from src.data_utils import DataHandler_Keras as dh
	from src.data_utils import DataHandlerLSTM as dhlstm
	from src.data_utils.plot_utils import *
	from src.data_utils import Support as sup
	from src.data_utils.Performance import *
	from src.data_utils.utils import *
	from src.data_utils.Recorder import Recorder as rec


# Define default args
model_name = "modelKerasRNN_arbitraryAgents"
exp_num = 20

root_folder = os.path.dirname(sys.path[0])
data_path = root_folder + '/data/'

# scenario = "2_agents/trajs/GA3C-CADRL-10"
scenario = "2_agents_swap/trajs/GA3C-CADRL-10-py27"
#scenario = "2_agents_random/trajs/GA3C-CADRL-10"


# Model directories
def parse_args():
	parser = argparse.ArgumentParser(description='LSTM model training')

	parser.add_argument('--model_name',
	                    help='Path to directory that comprises the model (default="model_name").',
	                    type=str, default=model_name)
	parser.add_argument('--num_test_sequences', help='Number of test sequences', type=int, default=5)
	parser.add_argument('--exp_num', help='Experiment number', type=int, default=exp_num)
	parser.add_argument('--n_samples', help='Number of samples', type=int, default=1)
	parser.add_argument('--scenario', help='Scenario of the dataset (default="").',
	                    type=str, default=scenario)
	parser.add_argument('--data_path', help='Path to directory that saves pickle data (default=" ").', type=str,default=data_path)
	parser.add_argument('--record', help='Is grid rotated? (default=True).', type=sup.str2bool,
	                    default=True)
	parser.add_argument('--save_figs', help='Save figures?', type=sup.str2bool,
	                    default=False)
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
	                    default=True)
	args = parser.parse_args()

	return args


test_args = parse_args()

if test_args.gpu:
	import tensorflow as tf
else:
	os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
	import tensorflow as tf

# cwd = os.getcwd()
# model_path = os.path.normpath(cwd + '/../') + '/trained_models/' + test_args.model_name + "/" + str(test_args.exp_num)

model_path = root_folder + '/trained_models/' + test_args.model_name + "/" + str(test_args.exp_num)

print("Loading data from: '{}'".format(model_path))
file = open(model_path + '/model_parameters.pkl', 'rb')
if sys.version_info[0] < 3:
	model_parameters = pkl.load(file)  # ,encoding='latin1')
else:
	model_parameters = pkl.load(file, encoding='latin1')
file.close()
args = model_parameters["args"]

# change some args because we are doing inference
args.truncated_backprop_length = 1
args.batch_size = 1
args.keep_prob = 1.0

# args.data_path = root_folder + '/data/2_agents_swap/trajs/'
# args.scenario = "GA3C-CADRL-10-py27"
# args.data_path = root_folder + '/data/2_agents/trajs/'
# args.scenario = "GA3C-CADRL-10"

# args.dataset = args.scenario+'.pkl'
# args.data_path = '../data/'

training_scenario = args.scenario
args.data_path = test_args.data_path
args.scenario = test_args.scenario
args.real_world_data = test_args.real_world_data
# args.dataset = '/' + args.scenario + '.pkl'
args.dataset = args.scenario + '.pkl'

data_prep = dh.DataHandler_Keras(args)
if test_args.unit_testing:
	data_prep.unit_test_data_()
assert os.path.isfile(args.data_path + args.dataset)
# Only used to create a map from png
map_args = {"file_name": 'map.png',
	            "resolution": 0.1,
	            "map_size": np.array([30., 30.]), }

data_prep.processData(**map_args)

# Import model
if "modelKerasRNN_arbitraryAgents" in args.model_name:
	if sys.version_info[0] < 3:
		from modelKerasRNN_arbitraryAgents import *
	else:
		from src.models.modelKerasRNN_arbitraryAgents import *

model = NetworkModel(args)
model.compile(loss=model.loss_object, optimizer=model.optimizer)

# Lists for logging of the input / output data of the model
input_list = []
grid_list = []
goal_list = []
ped_grid_list = []
y_ground_truth_list = []
y_pred_list = []  # uses ground truth as input at every step
other_agents_list = []
all_predictions = []
trajectories = []
batch_y = []
batch_loss = []

fig, axarr = pl.subplots(1, 2)
pl.show(block=False)


def clear_plot():
	axarr[0].clear()
	axarr[1].clear()
	axarr[0].set_title('Training Sample')
	axarr[1].set_title('Training Output')

# TODO: DataHandlerLSTM for Keras model has not been implemented yet
# if test_args.unit_testing:
# 	data_handler = dhlstm.DataHandlerLSTM(args)
# 	data_handler.unit_test_data_()


batch = data_prep.getBatch()
X_list = []
X_list.append(batch['vel'])
X_list.append( batch['other_agents_info'])

model.call(X_list)
model.load_weights(args.model_path + '/model_ckpt.h5')
if model.stateful: # Set model back to stateless for prediction
    for i in range(len(model.layers)):
        model.layers[i].stateful = False

# TODO: Convnet warmstart
# try:
# 		model.warmstart_convnet(args, sess)
# except:
	# print("")


for exp_idx in range(test_args.num_test_sequences):
	predictions = []

	# TODO: DataHandlerLSTM for Keras model has not been implemented yet
	# if test_args.unit_testing:
	# 	batch_x, batch_vel, batch_pos,batch_goal, batch_grid, other_agents_info, batch_target, other_agents_pos, traj = data_handler.getTestTrajectoryAsBatch(
	# 		exp_idx)  # trajectory_set random.randint(0, len(data_prep.dataset) - 1)
	# else:
	# 	batch_x, batch_vel, batch_pos,batch_goal, batch_grid, other_agents_info, batch_target, other_agents_pos, traj = data_prep.getTrajectoryAsBatch(
	# 		exp_idx)  # trajectory_set random.randint(0, len(data_prep.dataset) - 1)

	# batch_x, batch_vel, batch_pos,batch_goal, batch_grid, other_agents_info, batch_target, other_agents_pos, other_agents_vel, traj = data_prep.getTrajectoryAsBatch(exp_idx)  # trajectory_set random.randint(0, len(data_prep.dataset) - 1)
	new_batch = data_prep.getTrajectoryAsBatch(exp_idx,unit_testing=test_args.unit_testing)
	batch = deepcopy(new_batch)
	trajectories.append(batch['traj'])

	################## Custom code (begin) ##################
	# Sequence arrays must have shape [features, time steps]
	# vel_seq = np.reshape(batch['vel'], [batch['vel'].shape[1], batch['vel'].shape[2]])
	# vel_seq = np.concatenate( [np.zeros(vel_seq.shape[0], args.prev_horizon-1), vel_seq], axis=1 )
	# vel_input = dh.expand_sequence( vel_seq, args.prev_horizon) 

	# other_agents_states = np.concatenate([batch['other_agents_pos'], batch['other_agents_vel']], axis = 2)
	# other_agents_seq = np.reshape(other_agents_states, [other_agents_states.shape[1], other_agents_states.shape[2]])
	# other_agents_seq = np.concatenate( [np.zeros(other_agents_seq.shape[0], args.prev_horizon-1), other_agents_seq], axis=1 )
	# other_agents_input = dh.expand_sequence( other_agents_seq, args.prev_horizon) 

	# X_list = [vel_input, other_agents_input]

	# y_seq = np.reshape(batch['y'], [batch['y'].shape[1], batch['y'].shape[2]])
	# y_seq = np.concatenate( [np.zeros(y_seq.shape[0], args.prev_horizon-1), y_seq], axis=1 )
	# y_input = dh.expand_sequence( y_seq, args.prev_horizon) 


	################## Custom code (end) ##################


	x_input_series = np.zeros([1, args.input_dim])
	goal_input_series = np.zeros([1, 2])
	grid_input_series = np.zeros(
		[1, int(args.submap_width / args.submap_resolution), int(args.submap_height / args.submap_resolution)])
	ped_grid_series = np.zeros([1, args.pedestrian_vector_dim])
	y_ground_truth_series = np.zeros([1, args.prediction_horizon * 2])

	batch_y.append(batch['y'])

	for step in range(batch['x'].shape[0]):
		# Append to logging series
		x_input_series = np.append(x_input_series, np.expand_dims(batch['x'][step, 0, :],axis=0), axis=0)
		grid_input_series = np.append(grid_input_series, np.expand_dims(batch['grid'][step, :, :],axis=0), axis=0)
		goal_input_series = np.append(goal_input_series, np.expand_dims(batch['goal'][step, :],axis=0), axis=0)
		ped_grid_series = np.append(ped_grid_series, np.expand_dims(batch['other_agents_info'][step,0, :],axis=0), axis=0)
		y_ground_truth_series = np.append(y_ground_truth_series,batch['y'][step,:, :].reshape((1, args.prediction_horizon * 2)), axis=0)

	agents = batch['other_agents_info']

	X_list = []
	X_list.append( batch['vel'] )
	X_list.append( agents )

	y_model_pred = model.predict(X_list)

	for pred in y_model_pred:
		samples = [ np.reshape(pred, [1, pred.shape[0]*pred.shape[1]] ) ] # TODO: make it possible to deal with multiple samples (for variational models)
		predictions.append(samples)

	all_predictions.append(predictions)
	input_list.append(x_input_series[1:])
	goal_list.append(goal_input_series)
	grid_list.append(grid_input_series)
	ped_grid_list.append(ped_grid_series)
	y_ground_truth_list.append(y_ground_truth_series)
	other_agents_list.append(batch['other_agents_pos'])
	# update progress bar

if test_args.record:
		recorder = rec(args, data_prep.agent_container.occupancy_grid)
		#if test_args.real_world_data:
		#	print("Real data!!")
		#	recorder.plot_on_image(input_list, grid_list, ped_grid_list, all_predictions, y_ground_truth_list, other_agents_list,
		#	                       trajectories, args.rotated_grid, test_args.n_samples)
		#else:
		recorder.animate(input_list, grid_list, ped_grid_list, all_predictions, y_ground_truth_list, other_agents_list,
		                 trajectories,test_args)
		print("Recorder is done!")
else:
	"""Performance tests"""
	print("Average loss: " + str(np.mean(np.asarray(batch_loss))))
	pred_error, pred_error_summary_lstm = compute_trajectory_prediction_mse(args, trajectories, all_predictions)
	pred_fde, pred_error_summary_lstm_fde = compute_trajectory_fde(args, trajectories, all_predictions)
	args.scenario = training_scenario
	write_results_summary(np.mean(pred_error_summary_lstm), np.mean(pred_error_summary_lstm_fde), 0, args, test_args)
"""Quality tests

with tf.Session() as sess:

	model.warmstart_model(args, sess)

	trajectories.append(traj)

	model.reset_cells(data_prep.sequence_reset)

	# Test for 3 different headings
	theta = [-45*np.pi/180,0,45*np.pi/180]
	batch_x, batch_grid, other_agents_info, batch_target, other_agents_pos, traj = data_prep.getTrajectoryAsBatch(
		random.randint(0, len(data_prep.dataset) - 1))

	for step in range(batch_x.shape[1]):

		# Assemble feed dict for training
		for idx in range(len(theta)):
			x = np.array((1,1,2))
			x[0,0,0] = batch_x[:, step, 0]*np.cos(theta[idx])
			x[0, 0, 0] = batch_x[:, step, 0] * np.sin(theta[idx])

			agents = np.expand_dims(other_agents_info[:, step, :], axis=0)
			grid = np.expand_dims(batch_grid[:, step, :, :], axis=0)
			feed_dict_train = model.feed_test_dic(x, grid, agents)

			y_model_pred = model.predict(sess, feed_dict_train, False)

			predictions.append(np.squeeze(y_model_pred[0],axis=0))
		all_predictions.append(predictions)
"""

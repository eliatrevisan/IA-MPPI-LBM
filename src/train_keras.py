"""
Script to train Keras models.
Needs TensorFlow 2.
TODO: Implement a Keras version of DataHandlerLSTM
"""
import sys
import os
sys.path.append('../')
import numpy as np
import argparse
import pylab as pl
from multiprocessing.pool import ThreadPool
if sys.version_info[0] < 3:
	print("Using Python " + str(sys.version_info[0]))
	sys.path.append(sys.path[0] + '/data_utils') # sys.path.append('../src/data_utils')
	sys.path.append(sys.path[0] + '/models') # sys.path.append('../src/models')
	import DataHandler_Keras as dh
	import DataHandlerLSTM as dhlstm
	from plot_utils import *
	import Support as sup
	from utils import *
else:
	print("Using Python " + str(sys.version_info[0]))
	from src.data_utils import DataHandler_Keras as dh
	from src.data_utils import DataHandlerLSTM as dhlstm
	from src.data_utils.plot_utils import *
	from src.data_utils import Support as sup
	from src.data_utils.utils import *


import pickle as pkl
import time
from copy import deepcopy
from progressbar import progressbar

model_name = "modelKerasRNN_arbitraryAgents"
root_folder = os.path.dirname(sys.path[0])
model_path = root_folder + '/trained_models/' + model_name
pretrained_convnet_path = root_folder + "/trained_models/autoencoder_with_ped"
log_dir = model_path + '/log'

data_path = root_folder + '/data/'

scenario = '2_agents_swap/trajs/GA3C-CADRL-10'
# scenario_val = '2_agents_swap/trajs/GA3C-CADRL-10-py27'
scenario_val = '2_agents_random/trajs/GA3C-CADRL-10'

exp_num = 20

# Hyperparameters
n_steps = 20000

batch_size = 16
regularization_weight = 0.0001

# Time parameters
truncated_backprop_length = 10 # UNUSED, GETS REDEFINED AS PREV_HORIZON+1
prediction_horizon = 10
prev_horizon = 10

rnn_state_size = 64
rnn_state_size_lstm_grid = 128
rnn_state_size_lstm_ped = 64
rnn_state_size_bilstm_ped = 64
rnn_state_size_lstm_concat = 128
prior_size = 512
latent_space_size = 256
x_dim = 512
fc_hidden_unit_size = 128
learning_rate_init = 0.001
beta_rate_init = 0.01
keep_prob = 0.8
dropout = False
reg = 1e-4
n_mixtures = 0  # USE ZERO FOR MSE MODEL
grads_clip = 1.0
n_other_agents = 1
tensorboard_logging = False

# Model parameters
input_dim = 4  # [vx, vy]
input_state_dim = 2  # [vx, vy]
output_dim = 2  # data state dimension
output_pred_state_dim = 2  # vx, vy
pedestrian_vector_dim = 3
pedestrian_vector_state_dim = 2
cmd_vector_dim = 2
pedestrian_radius = 0.3
max_range_ped_grid = 10

print_freq = 2000
save_freq = 500
patience = 10
validate = False
dt = 0.4

warmstart_model = False
pretrained_convnet = False
pretained_encoder = False
multipath = False
real_world_data = False
end_to_end = True
agents_on_grid = False
rotated_grid = False
centered_grid = True
noise = False
normalize_data = False
real_world_data = False
regulate_log_loss = False

# Map parameters
submap_resolution = 0.1
submap_width = 6
submap_height = 6
diversity_update = False
predict_positions = False
warm_start_convnet = False

def parse_args():
	parser = argparse.ArgumentParser(description='LSTM model training')

	parser.add_argument('--model_name',
	                    help='Path to directory that comprises the model (default="model_name").',
	                    type=str, default=model_name)
	parser.add_argument('--model_path',
	                    help='Path to directory to save the model (default=""../trained_models/"+model_name").',
	                    type=str, default=model_path)
	parser.add_argument('--pretrained_convnet_path',
	                    help='Path to directory that comprises the pre-trained convnet model (default=" ").',
	                    type=str, default=pretrained_convnet_path)
	parser.add_argument('--log_dir',
	                    help='Path to the log directory of the model (default=""../trained_models/"+model_name").',
	                    type=str, default=log_dir)
	parser.add_argument('--scenario', help='Scenario of the dataset (default="").',
	                    type=str, default=scenario)
	parser.add_argument('--scenario_val', help='Dataset pkl file', type=str, default= scenario_val)
	parser.add_argument('--real_world_data', help='Real world dataset (default=True).', type=sup.str2bool,
	                    default=real_world_data)
	parser.add_argument('--data_path', help='Path to directory that saves pickle data (default=" ").', type=str,
	                    default=data_path)
	parser.add_argument('--dataset', help='Dataset pkl file', type=str, default= scenario + '.pkl')
	parser.add_argument('--dataset_val', help='Dataset pkl file', type=str, default= scenario_val + '.pkl')
	parser.add_argument('--data_handler', help='Datahandler class needed to load the data', type=str,
	                    default='LSTM')
	parser.add_argument('--warmstart_model', help='Restore from pretained model (default=False).', type=bool,
	                    default=warmstart_model)
	parser.add_argument('--warm_start_convnet', help='Restore from pretained convnet model (default=False).', type=bool,
	                    default=warm_start_convnet)
	parser.add_argument('--dt', help='Data samplig time (default=0.3).', type=float,
	                    default=dt)
	parser.add_argument('--n_steps', help='Number of epochs (default=10000).', type=int, default=n_steps)
	parser.add_argument('--batch_size', help='Batch size for training (default=32).', type=int, default=batch_size)
	parser.add_argument('--regularization_weight', help='Weight scaling of regularizer (default=0.01).', type=float,
	                    default=regularization_weight)
	parser.add_argument('--keep_prob', help='Dropout (default=0.8).', type=float,
	                    default=keep_prob)
	parser.add_argument('--learning_rate_init', help='Initial learning rate (default=0.005).', type=float,
	                    default=learning_rate_init)
	parser.add_argument('--beta_rate_init', help='Initial beta rate (default=0.005).', type=float,
	                    default=beta_rate_init)
	parser.add_argument('--dropout', help='Enable Dropout', type=sup.str2bool,
	                    default=dropout)
	parser.add_argument('--grads_clip', help='Gridient clipping (default=10.0).', type=float,
	                    default=grads_clip)
	parser.add_argument('--truncated_backprop_length', help='Backpropagation length during training (default=5).',
	                    type=int, default=truncated_backprop_length)
	parser.add_argument('--prediction_horizon', help='Length of predicted sequences (default=10).', type=int,
	                    default=prediction_horizon)
	parser.add_argument('--prev_horizon', help='Previous seq length.', type=int,
	                    default=prev_horizon)
	parser.add_argument('--rnn_state_size', help='Number of RNN / LSTM units (default=16).', type=int,
	                    default=rnn_state_size)
	parser.add_argument('--rnn_state_size_lstm_ped',
	                    help='Number of RNN / LSTM units of the pedestrian lstm layer (default=32).',
	                    type=int, default=rnn_state_size_lstm_ped)
	parser.add_argument('--rnn_state_size_bilstm_ped',
	                    help='Number of RNN / LSTM units of the pedestrian bidirectional lstm layer (default=32).',
	                    type=int, default=rnn_state_size_bilstm_ped)
	parser.add_argument('--rnn_state_size_lstm_grid',
	                    help='Number of RNN / LSTM units of the grid lstm layer (default=32).',
	                    type=int, default=rnn_state_size_lstm_grid)
	parser.add_argument('--rnn_state_size_lstm_concat',
	                    help='Number of RNN / LSTM units of the concatenation lstm layer (default=32).',
	                    type=int, default=rnn_state_size_lstm_concat)
	parser.add_argument('--prior_size', help='prior_size',
	                    type=int, default=prior_size)
	parser.add_argument('--latent_space_size', help='latent_space_size',
	                    type=int, default=latent_space_size)
	parser.add_argument('--x_dim', help='x_dim',
	                    type=int, default=x_dim)
	parser.add_argument('--fc_hidden_unit_size',
	                    help='Number of fully connected layer units after LSTM layer (default=64).',
	                    type=int, default=fc_hidden_unit_size)
	parser.add_argument('--input_state_dim', help='Input state dimension (default=).', type=int,
	                    default=input_state_dim)
	parser.add_argument('--input_dim', help='Input state dimension (default=).', type=float,
	                    default=input_dim)
	parser.add_argument('--output_dim', help='Output state dimension (default=).', type=float,
	                    default=output_dim)
	parser.add_argument('--output_pred_state_dim', help='Output prediction state dimension (default=).', type=int,
	                    default=output_pred_state_dim)
	parser.add_argument('--cmd_vector_dim', help='Command control dimension.', type=int,
	                    default=cmd_vector_dim)
	parser.add_argument('--n_mixtures', help='Number of modes (default=).', type=int,
	                    default=n_mixtures)
	parser.add_argument('--pedestrian_vector_dim', help='Number of angular grid sectors (default=72).', type=int,
	                    default=pedestrian_vector_dim)
	parser.add_argument('--pedestrian_vector_state_dim', help='Number of angular grid sectors (default=2).', type=int,
	                    default=pedestrian_vector_state_dim)
	parser.add_argument('--max_range_ped_grid', help='Maximum pedestrian distance (default=2).', type=float,
	                    default=max_range_ped_grid)
	parser.add_argument('--pedestrian_radius', help='Pedestrian radius (default=0.3).', type=float,
	                    default=pedestrian_radius)
	parser.add_argument('--n_other_agents', help='Number of other agents incorporated in the netwprk.', type=int,
	                    default=n_other_agents)
	parser.add_argument('--debug_plotting', help='Plotting for debugging (default=False).', type=int, default=0)
	parser.add_argument('--print_freq', help='Print frequency of training info (default=100).', type=int,
	                    default=print_freq)
	parser.add_argument('--save_freq', help='Save frequency of the temporary model during training. (default=20k).',
	                    type=int, default=save_freq)
	parser.add_argument('--exp_num', help='Experiment number', type=int, default=exp_num)
	parser.add_argument('--noise', help='Likelihood? (default=True).', type=sup.str2bool,
	                    default=False)
	parser.add_argument('--agents_on_grid', help='Likelihood? (default=True).', type=sup.str2bool,
	                    default=agents_on_grid)
	parser.add_argument('--normalize_data', help='Normalize? (default=False).', type=sup.str2bool,
	                    default=normalize_data)
	parser.add_argument('--rotated_grid', help='Rotate grid? (default=False).', type=sup.str2bool,
	                    default=rotated_grid)
	parser.add_argument('--centered_grid', help='Center grid? (default=False).', type=sup.str2bool,
	                    default=centered_grid)
	parser.add_argument('--sigma_bias', help='Percentage of the dataset used for trainning', type=float,
	                    default=0)
	parser.add_argument('--submap_width', help='width of occupancy grid', type=int, default=submap_width)
	parser.add_argument('--submap_height', help='height of occupancy grid', type=int, default=submap_height)
	parser.add_argument('--submap_resolution', help='Map resolution.', type=float,
	                    default=submap_resolution)
	parser.add_argument('--return_local_grid', help='Return local create. Used for speed.', type=sup.str2bool,
	                    default=False)
	parser.add_argument('--min_buffer_size', help='Minimum buffer size (default=1000).', type=int, default=1000)
	parser.add_argument('--max_buffer_size', help='Maximum buffer size (default=100k).', type=int, default=100000)
	parser.add_argument('--max_trajectories', help='maximum number of trajectories to be recorded', type=int, default=30)
	parser.add_argument('--end_to_end', help='End to end trainning.', type=sup.str2bool,
	                    default=False)
	parser.add_argument('--predict_positions', help='predict_positions.', type=sup.str2bool,
	                    default=predict_positions)
	parser.add_argument('--gpu', help='Enable GPU training.', type=sup.str2bool,
	                    default=True)
	parser.add_argument('--relative_info', help='Use relative info for other agents.', type=sup.str2bool,
	                    default=True)
	parser.add_argument('--regulate_log_loss', help='Enable GPU training.', type=sup.str2bool,
	                    default=regulate_log_loss)
	parser.add_argument('--diversity_update', help='diversity_update', type=sup.str2bool,
	                    default=diversity_update)
	parser.add_argument('--topics_config', help='yaml file containg subscription topics (default=" ").', type=str,
	                    default='../config/topics.yaml')
	parser.add_argument('--min_pos_x', help='min_pos_x', type=float, default=-1)
	parser.add_argument('--min_pos_y', help='min_pos_y', type=float, default=-1)
	parser.add_argument('--max_pos_x', help='max_pos_x', type=float, default=1)
	parser.add_argument('--max_pos_y', help='max_pos_y', type=float, default=1)
	parser.add_argument('--min_vel_x', help='min_vel_x', type=float, default=-1)
	parser.add_argument('--min_vel_y', help='min_vel_y', type=float, default=-1)
	parser.add_argument('--max_vel_x', help='max_vel_x', type=float, default=1)
	parser.add_argument('--max_vel_y', help='max_vel_y', type=float, default=1)
	parser.add_argument('--sx_vel', help='sx_vel', type=float, default=1)
	parser.add_argument('--sy_vel', help='sy_vel', type=float, default=1)
	parser.add_argument('--sx_pos', help='sx_pos', type=float, default=1)
	parser.add_argument('--sy_pos', help='sy_pos', type=float, default=1)
	args = parser.parse_args()

	return args

args = parse_args()

if args.gpu:
	import tensorflow as tf
else:
	os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
	import tensorflow as tf
tfk = tf.keras

args.model_path = root_folder + '/trained_models/' + args.model_name + "/" + str(args.exp_num)
save_path = args.model_path + '/model_ckpt.h5'
# args.dataset = '/' + args.scenario + '.pkl'
args.dataset = args.scenario + '.pkl' # Changed because previous version led to an error
args.dataset_val = args.scenario_val + '.pkl'
model_parameters = {"args": args}
print(args)
# Check whether model folder exists, otherwise make directory
if not os.path.exists(args.model_path):
	os.makedirs(args.model_path)
param_file = open(args.model_path + '/model_parameters.pkl', 'wb')
pkl.dump(model_parameters, param_file, protocol=2)  # encoding='latin1'
param_file.close()


data_prep = dh.DataHandler_Keras(args)
assert os.path.isfile(args.data_path + args.dataset)
# Only used to create a map from png
map_args = {"file_name": 'map.png',
	            "resolution": 0.1,
	            "map_size": np.array([30., 30.]), }

data_prep.processData(**map_args)

if os.path.isfile(args.data_path + args.dataset_val):
	validate = True
	val_args = args
	val_args.dataset = args.dataset_val
	data_prep_val = dh.DataHandler_Keras(val_args)
	data_prep_val.processData(**map_args)

# Import model
if "modelKerasRNN_arbitraryAgents" in args.model_name:
	if sys.version_info[0] < 3:
		from modelKerasRNN_arbitraryAgents import *
	else:
		from src.models.modelKerasRNN_arbitraryAgents import *


model = NetworkModel(args)
model.compile(loss=model.loss_object, optimizer=model.optimizer)

# fig, axarr = pl.subplots(1, 2)
# pl.show(block=False)

# def clear_plot():
# 	axarr[0].clear()
# 	axarr[1].clear()
# 	axarr[0].set_title('Training Sample')
# 	axarr[1].set_title('Training Output')


# Timing
avg_time_comp = 0.0
avg_time_total = 0.0

# TODO: Implement warmstart procedure for the Keras model
# if args.warmstart_model:
# 	model.warmstart_model(args, sess)
# else:
# 	# Initialize all TF variables
# 	sess.run(tf.global_variables_initializer())

# TODO: Implement warmstart procedure for the convnet in Keras
# try:
# 	if args.warm_start_convnet:
# 		model.warmstart_convnet(args, sess)
# except:
# 	print("Failed to initialized Convnet or Convnet does not exist")

# if the trainning was interrupted
try:
	first_step = int(open(args.model_path + "/tf_log", 'r').read().split('\n')[-2]) + 1
except:
	first_step = 1
epoch = 0
epoch_time = 0.0
patience_counter = 0
start_time = time.time()
best_loss = float("inf")
model.train_loss.reset_states()
model.val_loss.reset_states()

# Set up multithreadign for data handler
pool = ThreadPool(1)
res = None

for step in range(first_step, n_steps+1):
	# batch_x, batch_vel, batch_pos,batch_goal,batch_grid, batch_ped_grid, batch_y, other_agents_pos, other_agents_vel, new_epoch = data_prep.getBatch()
	print(step)
	if res == None:
		batch = data_prep.getBatch()
	else:
		batch = res.get(timeout=1)

	batch_backup = batch.copy()
	res = pool.apply_async(data_prep.getBatch)

	if batch_backup['new_epoch']:
		epoch += 1
		model.train_loss.reset_states()
		model.val_loss.reset_states()
		curr_time = time.time()
		epoch_time = curr_time - start_time
		start_time = curr_time

	# TODO: Implement custom cell reset method
	# Initialize the new sequences with a hidden state of zeros, the continuing sequences get assigned the previous hidden state
	# model.reset_cells(data_prep.sequence_reset)

	X_list = []
	X_list.append(batch_backup['vel'])
	# X_list.append( np.concatenate([batch['other_agents_pos']-batch['pos'], batch['other_agents_vel']], axis = -1) )
	X_list.append( batch_backup['other_agents_info'] ) # Only relative positions, velocities are absolute

	model.train_step(X_list, batch_backup['y'])

	if validate:
		batch_val = data_prep_val.getBatch()
		X_list_val = []
		X_list_val.append(batch_val['vel'])
		X_list_val.append( batch_val['other_agents_info'] ) # Only relative positions, velocities are absolute

		model.val_step(X_list_val, batch_val['y'])

	if step % save_freq == 0:
		if not validate:
			if model.train_loss.result() < best_loss:
				print("\nModel improved")
				model.save_weights(save_path)
				best_loss = model.train_loss.result()
				patience_counter = 0
			else:
				patience_counter += 1
		else:
			if model.val_loss.result() < best_loss:
				print("\nModel improved")
				model.save_weights(save_path)
				best_loss = model.val_loss.result()
				patience_counter = 0
			else:
				patience_counter += 1

	# Print training info
	if step % print_freq == 0:
		print('\n\nEpoch %d, Step %d, Epoch time: %f' % (epoch, step, epoch_time))
		print("Total steps: " + str(step))
		print("Train loss: %.4e" % (model.train_loss.result()))
		if validate:
			print("Validation loss: %.4e\n" % (model.val_loss.result()))

	if patience_counter >= patience:
		print("\nMaximum patience reached, stopping training early")
		break


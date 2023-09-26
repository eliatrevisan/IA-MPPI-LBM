import sys
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
sys.path.append('../')
import numpy as np
import argparse
import pylab as pl

if sys.version_info[0] < 3:
	print("Using Python " + str(sys.version_info[0]))
	sys.path.append('../src/data_utils')
	sys.path.append('../src/models')
	import DataHandler as dh
	import DataHandlerLSTM as dhlstm
	from plot_utils import *
	import Support as sup
	from utils import *
	from autoencoder_grid import ae_model as ae
else:
	print("Using Python " + str(sys.version_info[0]))
	from src.data_utils import DataHandler as dh
	from src.data_utils import DataHandlerLSTM as dhlstm
	from src.data_utils.plot_utils import *
	from src.data_utils import Support as sup
	from src.data_utils.utils import *
	from src.autoencoder_grid import ae_model as ae

import tensorflow as tf
import pickle as pkl
import time
from copy import deepcopy

tfk = tf.keras

model_name = "autoencoder_with_ped2"
model_path = '../trained_models/' + model_name
pretrained_convnet_path = "../trained_models/autoencoder_with_ped"
log_dir = model_path + '/log'
data_path = '../data/2_agents_swap/trajs/'
scenario = 'GA3C-CADRL-10-py27'
data_path = '../data/cyberzoo_experiments/'
scenario = 'all_trajectories'
exp_num = 6
data_path = '../data/'
scenario = '20_ped_with_obstacles/short_few_obstacles'

# Hyperparameters
n_epochs = 1200

batch_size = 16
regularization_weight = 0.0001

# Time parameters
truncated_backprop_length = 10
prediction_horizon = 10
prev_horizon = 0

rnn_state_size = 32
rnn_state_size_lstm_grid = 256
rnn_state_size_lstm_ped = 128
rnn_state_size_lstm_concat = 512
prior_size = 512
latent_space_size = 512
fc_hidden_unit_size = 256
learning_rate_init = 0.001
beta_rate_init = 0.01
keep_prob = 0.8
dropout = False
reg = 1e-4
n_mixtures = 3  # USE ZERO FOR MSE MODEL
grads_clip = 1.0
dropout = False
tensorboard_logging = False

# Model parameters
input_dim = 4  # [vx, vy]
input_state_dim = 2  # [vx, vy]
output_dim = 2  # data state dimension
output_pred_state_dim = 5  # ux uy simgax sigmay
pedestrian_vector_dim = 36
pedestrian_vector_state_dim = 2
cmd_vector_dim = 2
pedestrian_radius = 0.3
max_range_ped_grid = 10

print_freq = 2000
save_freq = 500
dt = 0.4

warmstart_model = False
pretrained_convnet = True
pretained_encoder = False
multipath = False
real_world_data = False
end_to_end = True
agents_on_grid = True
rotated_grid = False
centered_grid = True
noise = False
normalize_data = True
real_world_data = False

# Map parameters
submap_resolution = 0.1
submap_width = 6
submap_height = 6


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
	parser.add_argument('--real_world_data', help='Real world dataset (default=True).', type=sup.str2bool,
	                    default=real_world_data)
	parser.add_argument('--data_path', help='Path to directory that saves pickle data (default=" ").', type=str,
	                    default=data_path)
	parser.add_argument('--dataset', help='Dataset pkl file', type=str,
	                    default='/' + scenario + '.pkl')
	parser.add_argument('--data_handler', help='Datahandler class needed to load the data', type=str,
	                    default='LSTM')
	parser.add_argument('--warmstart_model', help='Restore from pretained model (default=False).', type=bool,
	                    default=warmstart_model)
	parser.add_argument('--dt', help='Data samplig time (default=0.3).', type=float,
	                    default=dt)
	parser.add_argument('--n_epochs', help='Number of epochs (default=10000).', type=int, default=n_epochs)
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
	                    help='Number of RNN / LSTM units of the grid lstm layer (default=32).',
	                    type=int, default=rnn_state_size_lstm_ped)
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
	parser.add_argument('--fc_hidden_unit_size',
	                    help='Number of fully connected layer units after LSTM layer (default=64).',
	                    type=int, default=fc_hidden_unit_size)
	parser.add_argument('--input_state_dim', help='Input state dimension (default=).', type=float,
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
	parser.add_argument('--min_buffer_size', help='Minimum buffer size (default=1000).', type=int, default=1000)
	parser.add_argument('--max_buffer_size', help='Maximum buffer size (default=100k).', type=int, default=100000)
	parser.add_argument('--max_trajectories', help='maximum number of trajectories to be recorded', type=int, default=30)
	parser.add_argument('--end_to_end', help='End to end trainning.', type=sup.str2bool,
	                    default=False)
	parser.add_argument('--topics_config', help='yaml file containg subscription topics (default=" ").', type=str,
	                    default='../config/topics.yaml')
	args = parser.parse_args()

	return args


args = parse_args()

args.model_path = '../trained_models/' + args.model_name + "/" + str(args.exp_num)

model_parameters = {"args": args}
print(args)
# Check whether model folder exists, otherwise make directory
if not os.path.exists(args.model_path):
	os.makedirs(args.model_path)
param_file = open(args.model_path + '/model_parameters.pkl', 'wb')
pkl.dump(model_parameters, param_file, protocol=2)  # encoding='latin1'
param_file.close()

if "GA3C" in args.scenario:
	data_prep = dh.DataHandler(args)
	# Collect data online
	if os.path.isfile(args.data_path + args.dataset):
		data_prep.load_data()
	else:
		data_prep.start_node()
else:
	data_prep = dhlstm.DataHandlerLSTM(args)
	# Only used to create a map from png
	map_args = {"file_name": 'map.png',
	            "resolution": 0.1,
	            "map_size": np.array([30., 30.]), }
	data_pickle = args.data_path + args.scenario + "/data" + str(prediction_horizon) + ".pickle"
	data_prep.processData(data_pickle, **map_args)

model = ae.ConvAEModel(learning_rate_init=learning_rate_init, grid_width=submap_width*10, grid_height=submap_height*10,
                         log_dir=log_dir, is_training=True, latent_space_dim=64)

config = tf.ConfigProto(
	device_count={'GPU': 0}
)

fig, axarr = pl.subplots(1, 2)
pl.show(block=False)


def clear_plot():
	axarr[0].clear()
	axarr[1].clear()
	axarr[0].set_title('Training Sample')
	axarr[1].set_title('Training Output')


with tf.Session(config=config) as sess:
	# Timing
	avg_time_comp = 0.0
	avg_time_total = 0.0

	if args.warmstart_model:
		model.warmstart_model(args, sess)
	else:
		# Initialize all TF variables
		sess.run(tf.global_variables_initializer())
	model.warmstart_convnet(args, sess)

	# if the trainning was interrupted
	try:
		step = int(open(args.model_path + "/tf_log", 'r').read().split('\n')[-2]) + 1
	except:
		step = 1
	epoch = 0
	training_loss = []
	diversity_loss = []
	while epoch <= n_epochs:
		start_time_loop = time.time()
		# if res == None:
		batch_x, batch_vel, batch_grid, batch_ped_grid, batch_y, other_agents_pos, new_epoch = data_prep.getBatch()

		if new_epoch:
			epoch += 1

		data_prep.add_other_agents_to_grid(batch_grid, batch_x, other_agents_pos, args)

		feed_dict_train = {model.input_placeholder: batch_grid[:, 0, :, :],
		                   model.output_placeholder: batch_grid[:, 0, :, :]}
		time_start = time.time()
		_, total_loss, deconv1, summary_str = sess.run([model.update, model.total_loss, model.deconv1, model.summary],
		                                               feed_dict=feed_dict_train)

		model.summary_writer.add_summary(summary_str,   step)
		time_avg += time.time() - time_start
		if step % print_freq == 0:
			clear_plot()
			sup.plotGrid(batch_grid[0, 0, :, :], axarr[0, 0], color='k', alpha=0.5)
			sup.plotGrid(deconv1[0, :, :], axarr[0, 1], color='k', alpha=0.5)
			fig.canvas.draw()
			time_avg = 0.0

		# Save model every now and then
		if step % save_freq == 0:
			save_path = model_path + '/model_ckpt'
			model.saver.save(sess, save_path, global_step=  step)
			print('Epoch {} Step {}: Saving model under {}'.format(epoch,   step, save_path))
			full_path = model_path + '/final-model.ckpt'
			model.saver.save(sess, full_path)
			print('Saved final model under "{}"'.format(full_path))

		step +=1

sess.close()

print("test")

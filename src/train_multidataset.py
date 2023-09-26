import sys
import os
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
else:
	print("Using Python " + str(sys.version_info[0]))
	from src.data_utils import DataHandler as dh
	from src.data_utils import DataHandlerLSTM as dhlstm
	from src.data_utils.plot_utils import *
	from src.data_utils import Support as sup
	from src.data_utils.utils import *


import pickle as pkl
import time
from copy import deepcopy

model_name = "VDGNN_multihead"
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
n_epochs = 2

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
latent_space_size = 256
x_dim = 512
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
normalize_data = False
real_world_data = False
regulate_log_loss = False
# Map parameters
submap_resolution = 0.1
submap_width = 6
submap_height = 6
diversity_update = False
predict_positions = False

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
	parser.add_argument('--predict_positions', help='predict_positions.', type=sup.str2bool,
	                    default=predict_positions)
	parser.add_argument('--gpu', help='Enable GPU training.', type=sup.str2bool,
	                    default=False)
	parser.add_argument('--relative_info', help='Use relative info for other agents.', type=sup.str2bool,
	                    default=False)
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
	data_pickle = args.data_path+args.scenario + "/data" + str(args.prediction_horizon) +"_"+str(args.truncated_backprop_length)+".pickle"
	data_prep.processData(data_pickle,**map_args)

	original_scenario = args.scenario
	args.scenario = "ewap_dataset/seq_eth"
	data_prep_2 = dhlstm.DataHandlerLSTM(args)
	data_pickle = args.data_path + args.scenario + "/data_seq_eth" + str(prediction_horizon) +"_"+str(args.truncated_backprop_length)+ ".pickle"
	data_prep_2.processData(data_pickle, **map_args)

	args.scenario = "st"
	data_prep_3 = dhlstm.DataHandlerLSTM(args)
	data_pickle = args.data_path + args.scenario + "/data_" +args.scenario + str(prediction_horizon)+"_"+str(args.truncated_backprop_length) + ".pickle"
	data_prep_3.processData(data_pickle, **map_args)

	args.scenario = "zara_02"
	data_prep_4 = dhlstm.DataHandlerLSTM(args)
	data_pickle = args.data_path + args.scenario + "/data_" +args.scenario + str(prediction_horizon)+"_"+str(args.truncated_backprop_length) + ".pickle"
	data_prep_4.processData(data_pickle, **map_args)

	args.scenario = original_scenario
# Import model
if "noGrid" in args.model_name:
	if sys.version_info[0] < 3:
		from VGDNN_noGrid import *
	else:
		from src.models.VGDNN_noGrid import *
elif "Grid" in args.model_name:
	if sys.version_info[0] < 3:
		from VGDNN_Grid import *
	else:
		from src.models.VGDNN_Grid import *
elif "RNN" in args.model_name:
	if sys.version_info[0] < 3:
		from RNN import *
	else:
		from src.models.RNN import *
elif "KL" in args.model_name:
	if sys.version_info[0] < 3:
		from VGDNN_KL import *
	else:
		from src.models.VGDNN_KL import *
elif "diversity" in args.model_name:
	if sys.version_info[0] < 3:
		from VGDNN_attention_diversity import *
	else:
		from src.models.VGDNN_attention_diversity import *
elif "ped" in args.model_name:
	if sys.version_info[0] < 3:
		from VGDNN_ped_attention import *
	else:
		from src.models.VGDNN_ped_attention import *
elif "attention" in args.model_name:
	if sys.version_info[0] < 3:
		from VGDNN_attention import *
	else:
		from src.models.VGDNN_attention import *
elif "multihead" in args.model_name:
	if sys.version_info[0] < 3:
		from VGDNN_multihead_attention import *
	else:
		from src.models.VGDNN_multihead_attention import *
elif "simple" in args.model_name:
	if sys.version_info[0] < 3:
		from VGDNN_simple import *
	else:
		from src.models.VGDNN_simple import *
elif "pos" in args.model_name:
	if sys.version_info[0] < 3:
		from VGDNN_pos import *
	else:
		from src.models.VGDNN_pos import *
else:
	if sys.version_info[0] < 3:
		from VGDNN import *
	else:
		from src.models.VGDNN import *

model = NetworkModel(args)

config = tf.ConfigProto(
        device_count = {'GPU': 0}
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
	dataset = random.randint(0, 3)
	while step <= 20000:
		start_time_loop = time.time()
		# if res == None:

		if dataset == 0:
			batch_x, batch_vel, batch_pos, batch_grid, batch_ped_grid, batch_y, other_agents_pos, new_epoch = data_prep.getBatch()
		if dataset == 1:
			batch_x, batch_vel, batch_pos,batch_grid, batch_ped_grid, batch_y, other_agents_pos, new_epoch = data_prep_2.getBatch()
		if dataset == 2:
			batch_x, batch_vel,batch_pos, batch_grid, batch_ped_grid, batch_y, other_agents_pos, new_epoch = data_prep_3.getBatch()
		if dataset == 3:
			batch_x, batch_vel, batch_pos, batch_grid, batch_ped_grid, batch_y, other_agents_pos, new_epoch = data_prep_4.getBatch()

		if new_epoch:
			dataset = random.randint(0, 3)
			epoch += 1

		# Initialize the new sequences with a hidden state of zeros, the continuing sequences get assigned the previous hidden state
		model.reset_cells(data_prep.sequence_reset)

		# Backup variables to plot
		if args.input_state_dim <4:
			batch_x_backup = deepcopy(batch_vel)
		else:
			batch_x_backup = deepcopy(batch_x)
		batch_grid_backup = deepcopy(batch_grid)
		batch_ped_grid_backup = deepcopy(batch_ped_grid)
		if args.predict_positions:
			batch_y_backup = deepcopy(batch_pos)
		else:
			batch_y_backup = deepcopy(batch_y)
		other_agents_pos_backup = other_agents_pos

		start_time_comp = time.time()

		if step > 2000 and args.diversity_update:
			model.update_test_hidden_state()
			feed_test_dic = model.feed_test_dic(batch_x_backup, batch_grid_backup, batch_ped_grid_backup, batch_y_backup)
			y_model_pred, loss, output_decoder = model.predict(sess, feed_test_dic, False)
			batch_y_varible = np.zeros((args.batch_size,args.truncated_backprop_length,args.prediction_horizon*args.output_dim))
			for t_id in range(len(y_model_pred)):
				for batch_id in range(y_model_pred[0].shape[0]):
					for pred_id in range(args.prediction_horizon):
						new_batch_x = y_model_pred[t_id][batch_id][pred_id*args.output_pred_state_dim*args.n_mixtures]
						new_batch_y = y_model_pred[t_id][batch_id][pred_id * args.output_pred_state_dim * args.n_mixtures]
						batch_y_varible[batch_id,t_id,2*pred_id:2*pred_id+args.output_dim] = np.array([new_batch_x,new_batch_y])
					# Assemble feed dict for training
			feed_dict_train = model.feed_dic(batch_x_backup, batch_grid_backup, batch_ped_grid_backup, step,
					                                 batch_y_backup,batch_y_varible)
		else:
			batch_y_varible = batch_y_backup.copy()
			try:
				if "diversity" in args.model_name:
					feed_dict_train = model.feed_dic(batch_x_backup, batch_grid_backup, batch_ped_grid_backup, step,
					                                 batch_y_backup, batch_pos,batch_y_varible)
				else:
					feed_dict_train = model.feed_dic(batch_x_backup, batch_grid_backup, batch_ped_grid_backup, step,
					                                 batch_y_backup, batch_pos)
			except:
				feed_dict_train = model.feed_dic(batch_x_backup, batch_grid_backup, batch_ped_grid_backup, step,
				                                 batch_y_backup)

		batch_loss, kl_loss, _model_prediction, _summary_str, lr, beta, output_decoder, div_loss, autoencoder_loss = model.run(
			sess, feed_dict_train)

		if np.mean(autoencoder_loss) > 0.01:
			print("Autoencoder loss: " +str(np.mean(autoencoder_loss)))
			model.run_autoencoder(sess, feed_dict_train)
			model.convnet_saver.save(sess, args.pretrained_convnet_path+ '/final-model.ckpt')

		diversity_loss.append(div_loss)
		training_loss.append(batch_loss)

		avg_time_comp += time.time() - start_time_comp
		avg_time_total += time.time() - start_time_loop

		if tensorboard_logging:
			model.summary_writer.add_summary(_summary_str, step)

		# Print training info
		if step % print_freq == 0:
			"""
			for batch_id in range(batch_grid_backup.shape[0]):
				input_grid = batch_grid_backup[batch_id, 0, :, :]
				output_grid = output_decoder[batch_id, :, :, 0]
				clear_plot()
				sup.plotGrid(batch_grid_backup[batch_id, 0, :, :], axarr[0], color='k', alpha=0.5)
				sup.plotGrid(output_grid, axarr[1], color='r', alpha=0.5)
				fig.canvas.draw()
				pl.show(block=False)
			#plot_batch_OpenCV(step, batch_x, batch_grid, batch_ped_grid, batch_y, other_agents_pos, _model_prediction, args)

			# Testing on a trajectory from a different dataset
			"""
			batch_x, batch_vel, batch_pos,batch_grid, other_agents_info, batch_target, other_agents_pos, traj = data_prep.getTrajectoryAsBatch(
				random.randint(0, len(data_prep.trajectory_set) - 1))  # trajectory_set random.randint(0, len(data_prep.dataset) - 1)

			model.reset_test_cells(np.ones((args.batch_size)))
			predictions = []
			for traj_step in range(batch_x.shape[1]):
				# Assemble feed dict for training
				if args.input_state_dim < 4:
					x = np.repeat(np.repeat(np.expand_dims(batch_vel[:, traj_step, :], axis=0),args.truncated_backprop_length,axis=1),args.batch_size,axis=0)
				else:
					x = np.repeat(np.repeat(np.expand_dims(batch_x[:, traj_step, :], axis=0),args.truncated_backprop_length,axis=1),args.batch_size,axis=0)
				agents = np.repeat(np.repeat(np.expand_dims(other_agents_info[:, traj_step, :], axis=0),args.truncated_backprop_length,axis=1),args.batch_size,axis=0)
				grid = np.repeat(np.repeat(np.expand_dims(batch_grid[:, traj_step, :, :], axis=0),args.truncated_backprop_length,axis=1),args.batch_size,axis=0)
				y = np.repeat(np.repeat(np.expand_dims(batch_target[:, traj_step, :], axis=0),args.truncated_backprop_length,axis=1),args.batch_size,axis=0)
				pos = np.repeat(np.repeat(np.expand_dims(batch_pos[:, traj_step, :], axis=0),args.truncated_backprop_length,axis=1),args.batch_size,axis=0)

				try:
					feed_dict_ = model.feed_test_dic(x, grid, agents, y,pos)
				except:
					feed_dict_ = model.feed_test_dic(x, grid, agents, y)

				y_model_pred, loss, output_decoder = model.predict(sess, feed_dict_, True)

				predictions.append([y_model_pred[0]])

			plot_scenario_vel_OpenCV_simdata([traj], [batch_target], [predictions], args,step)
			plot_local_scenario_vel_OpenCV([traj], [batch_target], [predictions], args,step)

			print(bcolors.WARNING + 'Epoch {0}: batch_loss {1:.5f} \t comp time.: {2:.5f}ms \t tot. time.: {3:.5f}ms'.format(
				epoch,
				batch_loss,
				avg_time_comp * 1000 / print_freq,
				avg_time_total * 1000 / print_freq) + bcolors.ENDC)

			print("Step: " + str(step) + " Loss: " + str(batch_loss) + " KL Loss: " + str(kl_loss) +" Beta: " + str(beta) + " Diversity: " + str(np.mean(diversity_loss))+ " Autoencoder loss: " + str(np.mean(autoencoder_loss)))

			with open(args.model_path + "/tf_log", 'a') as f:
				f.write(str(step) + '\n')
			save_path = args.model_path + '/model_ckpt'
			model.full_saver.save(sess, save_path, global_step=step)

			print('Step {}: Saving model under {}'.format(step, save_path))

			print('Epoch {0},Step {0}: Learning_rate {1:.5f} \t'.format(epoch, lr))
			print("Total steps: " + str(step))

		step = step + 1

	write_summary(training_loss[-1], args)
	full_path = args.model_path + '/final-model.ckpt'
	model.full_saver.save(sess, full_path)
	print(bcolors.OKBLUE + 'Final loss: batch_loss ' + str(training_loss[-1]) + bcolors.ENDC)
	print('Saved final model under "{}"'.format(full_path))

	if tensorboard_logging:
		model.summary_writer.close()

sess.close()

print("test")

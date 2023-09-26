"""
Script to train Keras models.
Needs TensorFlow 2.
TODO: Implement a Keras version of DataHandlerLSTM
"""
import sys
import os
sys.path.append('../')
import numpy as np
import pylab as pl
if sys.version_info[0] < 3:
	print("Using Python " + str(sys.version_info[0]))
	sys.path.append(sys.path[0] + '/data_utils') # sys.path.append('../src/data_utils')
	sys.path.append(sys.path[0] + '/models') # sys.path.append('../src/models')
	import DataHandler_Keras_v2 as dh
	import DataHandlerLSTM as dhlstm
	from plot_utils import *
	import Support as sup
	from utils import *
else:
	print("Using Python " + str(sys.version_info[0]))
	from src.data_utils import DataHandler_Keras_v2 as dh
	from src.data_utils import DataHandlerLSTM as dhlstm
	from src.data_utils.plot_utils import *
	from src.data_utils import Support as sup
	from src.data_utils.utils import *


import pickle as pkl
import time
from copy import deepcopy
from progressbar import progressbar

model_name = "modelKerasRNN_arbitraryAgents"
input1_type = "polarvel"
input2_type = "posrel_velabs"
exp_num = 205

n_steps = 20000
print_freq = 2000
save_freq = 500
patience = 10

scenario = '2_agents/trajs/GA3C-CADRL-10'
# scenario_val = '2_agents_swap/trajs/GA3C-CADRL-10-py27'
# scenario_val = '2_agents_random/trajs/GA3C-CADRL-10'
scenario_val = 'none'

root_folder = os.path.dirname(sys.path[0])
model_path = root_folder + '/trained_models/' + model_name
pretrained_convnet_path = root_folder + "/trained_models/autoencoder_with_ped"
log_dir = model_path + '/log'
data_path = root_folder + '/data/'

defaults = {
    "model_name": model_name,
    "input1_type":input1_type,
    "input2_type":input2_type,
    "exp_num": exp_num,
    "scenario":scenario,
    "scenario_val":scenario_val,
    "model_path": model_path,
    "pretrained_convnet_path": pretrained_convnet_path,
    "log_dir": log_dir,
    "data_path": data_path,
    "n_steps": n_steps,
    "print_freq": print_freq,
    "save_freq": save_freq,
    "patience": patience,

    # Time parameters
    "truncated_backprop_length": 10, # UNUSED, GETS REDEFINED AS PREV_HORIZON+1
    "prediction_horizon": 10,
    "prev_horizon": 10,

    # Hyperparameters
    "batch_size": 16,
    "regularization_weight": 0.0001,
    "rnn_state_size": 64,
    "rnn_state_size_lstm_grid": 128,
    "rnn_state_size_lstm_ped": 64,
    "rnn_state_size_bilstm_ped": 64,
    "rnn_state_size_lstm_concat": 128,
    "prior_size": 512,
    "latent_space_size": 256,
    "x_dim": 512,
    "fc_hidden_unit_size": 128,
    "learning_rate_init": 0.001,
    "beta_rate_init": 0.01,
    "keep_prob": 0.8,
    "dropout": False,
    "reg": 1e-4,
    "n_mixtures": 0,  # USE ZERO FOR MSE MODEL
    "grads_clip": 1.0,

    "tensorboard_logging": False,

    # Model parameters
    "input_dim": 4,  # [vx, vy]
    "input_state_dim": 2,  # [vx, vy]
    "output_dim": 2,  # data state dimension
    "output_pred_state_dim": 2,  # vx, vy
    "pedestrian_vector_dim": 4,
    "pedestrian_vector_state_dim": 2,
    "cmd_vector_dim": 2,
    "pedestrian_radius": 0.3,
    "max_range_ped_grid": 10,
    "dt": 0.4,

    "warmstart_model": False,
    "pretrained_convnet": False,
    "pretained_encoder": False,
    "multipath": False,
    "real_world_data": False,
    "end_to_end": True,
    "agents_on_grid": False,
    "rotated_grid": False,
    "centered_grid": True,
    "noise": False,
    "normalize_data": False,
    "real_world_data": False,
    "regulate_log_loss": False,

    # Map parameters
    "submap_resolution": 0.1,
    "submap_width": 6,
    "submap_height": 6,
    "diversity_update": False,
    "predict_positions": False,
    "warm_start_convnet": False
}

args = parse_args(defaults, "Train")

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

validate = False
if ("GA3C" in args.scenario) or("RVO" in args.scenario): # ("GA3C" in args.scenario) or
	data_prep = dh.DataHandler(args)
	# Collect data online
	# if os.path.isfile(args.data_path + args.dataset):
	# 	data_prep.load_data()
	# else:
	# 	data_prep.start_node()
	assert os.path.isfile(args.data_path + args.dataset)
	data_prep.load_data()

	if os.path.isfile(args.data_path + args.dataset_val):
		print("Performing validation with dataset: " + args.dataset_val)
		validate = True
		val_args = args
		val_args.dataset = args.dataset_val
		data_prep_val = dh.DataHandler(val_args)
		data_prep_val.load_data()
	else: 
		print("No validation")

else:
	data_prep = dhlstm.DataHandlerLSTM(args)
	# Only used to create a map from png
	map_args = {"file_name": 'map.png',
	            "resolution": 0.1,
	            "map_size": np.array([30., 30.]), }
	data_pickle = args.data_path+args.scenario + "/data" + str(args.prediction_horizon) +"_"+str(args.truncated_backprop_length)+".pickle"
	data_prep.processData(data_pickle,**map_args)
	data_prep.calc_scale()
	args.min_pos_x = data_prep.min_pos_x
	args.min_pos_y = data_prep.min_pos_y
	args.max_pos_x = data_prep.max_pos_x
	args.max_pos_y = data_prep.max_pos_y
	args.min_vel_x = data_prep.min_vel_x
	args.min_vel_y = data_prep.min_vel_y
	args.max_vel_x = data_prep.max_vel_x
	args.max_vel_y = data_prep.max_vel_y
	args.sx_vel = data_prep.sx_vel
	args.sy_vel = data_prep.sy_vel
	args.sx_pos = data_prep.sx_pos
	args.sy_pos = data_prep.sy_pos
	traj_len=0
	for traj_id in range(len(data_prep.trajectory_set)):
		traj_len += data_prep.trajectory_set[traj_id][1].pose_vec.shape[0]

	traj_len = traj_len/len(data_prep.trajectory_set)

# Import model
model = model_selector(args)
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

for step in progressbar(range(first_step, args.n_steps+1)):
	batch = data_prep.getBatch()

	if batch['new_epoch']:
		epoch += 1
		model.train_loss.reset_states()
		model.val_loss.reset_states()
		curr_time = time.time()
		epoch_time = curr_time - start_time
		start_time = curr_time

	model.train_step(batch['input'], batch['target'])

	if validate:
		batch_val = data_prep_val.getBatch()
		model.val_step(batch_val['input'], batch_val['target'])

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
		print("\n")

	if patience_counter >= patience:
		print("\nMaximum patience reached, stopping training early")
		break

write_keras_results_summary(float(best_loss), args)
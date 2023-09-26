import sys
import os

sys.path.append('../')
import numpy as np
import argparse
import pylab as pl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import pickle as pkl
# import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import random
import progressbar
import scipy.io as sio
import json
import importlib
from copy import deepcopy
if sys.version_info[0] < 3:
	sys.path.append('../src/data_utils')
	sys.path.append('../src/models')
	import DataHandlerLSTM as dhlstm
	from plot_utils import *
	import Support as sup
	from Performance import *
	from utils import *
	import Recorder as rec
	from social_force_model import ForceModel
else:
	from src.data_utils import DataHandlerLSTM as dhlstm
	from src.data_utils.plot_utils import *
	from src.data_utils import Support as sup
	from src.data_utils.Performance import *
	from src.data_utils.utils import *
	from src.data_utils.Recorder import Recorder as rec
	from src.simulators.social_force_model import *
	import src.data_utils.Trajectory as traj


# Model directories
def parse_args():
	parser = argparse.ArgumentParser(description='LSTM model training')

	parser.add_argument('--model_name',
	                    help='Path to directory that comprises the model (default="model_name").',
	                    type=str, default="VGDNN_simple")
	parser.add_argument('--num_test_sequences', help='Number of test sequences', type=int, default=10)
	parser.add_argument('--exp_num', help='Experiment number', type=int, default=9)
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
	args = parser.parse_args()

	return args


test_args = parse_args()

if test_args.gpu:
	import tensorflow as tf
else:
	os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
	import tensorflow as tf

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

# change some args because we are doing inference
args.truncated_backprop_length = 1
args.batch_size = 1
args.keep_prob = 1.0

training_scenario = args.scenario
args.scenario = test_args.scenario
args.real_world_data = test_args.real_world_data
args.dataset = '/' + args.scenario + '.pkl'
data_prep = dhlstm.DataHandlerLSTM(args)
# Only used to create a map from png
map_args = {"file_name": 'map.png',
	            "resolution": 0.1,
	            "map_size": np.array([40., 7.]), }

data_prep.processData(**map_args)

data_path =args.data_path + args.scenario
grid = load_map(data_path)
sf_model = ForceModel(prediction_steps=12, occ_grid=grid,ped_num=2)

# Import model
mod = importlib.import_module("src.models." + args.model_name)
print("Importing : " + args.model_name)
model = mod.NetworkModel(args)

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

config = tf.ConfigProto(
	device_count={'GPU': 0}
)


def clear_plot():
	axarr[0].clear()
	axarr[0].set_title('Training Sample')

if test_args.unit_testing:
	data_handler = dhlstm.DataHandlerLSTM(args)
	data_handler.unit_test_data_(map_args)

with tf.Session(config=config) as sess:
	model.warmstart_model(args, sess)
	try:
			model.warmstart_convnet(args, sess)
	except:
		print("")

	initial_poses = [np.array([[-15.0, 0], [-12.0, 0]]), np.array([[-12.0, 0], [-15.0, 0]]), np.array([[15.0, 0], [-15.0, 0]])]
	initial_velocities = [np.array([[1.0, 0], [1.0, 0]]),np.array([[1.0, 0], [1.0, 0]]),np.array([[-1.0, 0], [1.0, 0]])]
	all_goals = [np.array([[15.0, 0], [15.0, 0]]),np.array([[15.0, 0], [15.0, 0]]),np.array([[-15.0, 0], [15.0, 0]])]

	for exp_idx in range(len(initial_poses)):

		initial_pose = initial_poses[exp_idx]
		initial_velocity = initial_velocities[exp_idx]
		goals = all_goals[exp_idx]
		next_poses = initial_pose
		next_speed = initial_velocity
		trajectory = traj.Trajectory(goal=goals[0])
		pose = np.zeros([1, 3])
		speed = np.zeros([1, 3])
		pose[0, :2] = initial_pose[0]
		speed[0, :2] = initial_velocity[0]
		trajectory.addData(0, pose, speed, goals[0])
		current_pos = next_poses[0]
		current_vel = next_speed[0]

		# Variables for Post-Processing
		predictions = []
		x_input_series = np.concatenate((np.expand_dims(current_pos,axis=0),np.expand_dims(current_vel,axis=0)),axis=1)
		goal_input_series = np.zeros([1, 2])
		y_ground_truth_series = np.zeros([1, args.prediction_horizon * 2])
		other_agents = []
		y_pred_series = np.zeros([1, args.n_mixtures * args.prediction_horizon * args.output_pred_state_dim])
		grid_input_series = np.zeros([1, 1,60,60])
		ped_grid_series = np.zeros([1,1,args.n_other_agents,args.pedestrian_vector_dim*args.prediction_horizon])
		model.reset_test_cells(np.ones((args.batch_size)))
		cell_state_list= []
		cell_ped_list = []
		cell_concat_list = []

		sf_model.set_initial_pose(next_poses, next_speed, goals)

		for step in range(50):
			samples = []
			if step > 0:
				x_input_series = np.append(x_input_series, np.concatenate(
					(np.expand_dims(current_pos, axis=0), np.expand_dims(current_vel, axis=0)), axis=1), axis=0)

			# Assemble feed dict for training

			x = np.reshape(next_speed[0],(1,1,-1))
			agents = np.zeros([1,1,args.n_other_agents,args.pedestrian_vector_dim*args.prediction_horizon])
			# agent zero is the predicting agent
			current_pos = next_poses[0]
			current_vel = next_speed[0]
			for pred_step in range(args.prediction_horizon):
					for ag_id in range(min([next_poses.shape[0]-1, args.n_other_agents])):
						next_pose = np.expand_dims(next_poses[ag_id+1, :2] + args.dt * next_speed[ag_id+1, :2] * (pred_step + 1),axis=0)
						next_pose_rotated = sup.positions_in_local_frame(current_pos, 0, next_pose)
						agents[0, 0, ag_id, args.pedestrian_vector_dim * pred_step:args.pedestrian_vector_dim * pred_step+2] = \
									next_pose_rotated[0,:]#np.tanh(/ (np.abs(dx) + 0.1)
						velocity_rotated = sup.positions_in_local_frame(current_vel, 0, np.expand_dims(next_speed[ag_id+1, :2],axis=0))
						agents[0, 0, ag_id, args.pedestrian_vector_dim * pred_step + 2:args.pedestrian_vector_dim * pred_step + 4] = \
									velocity_rotated[0,:]
					for ag_id in range(min([next_poses.shape[0]-1, args.n_other_agents]), args.n_other_agents):
						agents[0, 0, ag_id, args.pedestrian_vector_dim * pred_step] = -1
						agents[0, 0, ag_id, args.pedestrian_vector_dim * pred_step + 1] = -1
						agents[0, 0, ag_id, args.pedestrian_vector_dim * pred_step + 2] = 0
						agents[0, 0, ag_id, args.pedestrian_vector_dim * pred_step + 3] = 0

			grid = np.expand_dims(np.expand_dims(data_handler.agent_container.occupancy_grid.getSubmapByCoords(current_pos[0],
																																			 current_pos[1],
																																			 args.submap_width,
																																			 args.submap_height),axis=0),axis=0)

			feed_dict_ = model.feed_test_dic(x, grid, agents, 0)

			# Append to logging series
			if step == 0:
				grid_input_series = grid[0]
			else:
				grid_input_series = np.append(grid_input_series, grid[0], axis=0)
			ped_grid_series = np.append(ped_grid_series, agents, axis=0)

			y_model_pred, output_decoder, cell_outputs_series_state, cell_outputs_series_lstm_ped, cell_concat = model.predict(sess, feed_dict_, True)

			pred_poses, pred_velocities = sf_model.predict(0, next_poses[0], next_speed[0], goals[0], args.dt)

			next_poses, next_speed = sf_model.simulation_step(args.dt)

			# does not work yet to propagate the dynamics of the network
			#next_poses[0,:] = x_input_series[step,:2] + args.dt*y_model_pred[0][0,:2]
			#next_speed[0,:] = args.dt*y_model_pred[0][0,:2]

			y_ground_truth_series = np.append(y_ground_truth_series, np.reshape(pred_velocities,newshape=(1,-1)), axis=0)
			other_agents.append(deepcopy(next_poses[1:]))
			pose[0, :2] = next_poses[0]
			speed[0, :2] = next_speed[0]
			cell_state_list.append(cell_outputs_series_state[0][0,:])
			cell_ped_list.append(cell_outputs_series_lstm_ped[0][0, :])
			cell_concat_list.append(cell_concat[0][0, :])
			samples.append(y_model_pred[0])

			# append data sample to trajectory (timestamp in [ns], pose and vel as np 1x3 arrays)
			trajectory.addData((step+1)*args.dt, pose, speed, goals[0])

			"""
			output_grid = output_decoder[0, :, :, 0]
			clear_plot()
			sup.plotGrid(batch_grid[0, step, :, :], axarr[0], color='k', alpha=0.5)
			sup.plotGrid(output_grid, axarr[1], color='r', alpha=0.5)
			fig.canvas.draw()
			pl.show(block=False)
			"""

			predictions.append(samples)

		latent_analysis = False
		if latent_analysis:

			fig = pl.figure()
			ax = pl.axes(projection="3d")
			pl.show(block=False)
			x_state = np.arange(0, args.rnn_state_size, 1)
			x_ped = np.arange(0, args.rnn_state_size_lstm_ped * 2, 1)
			x_concat = np.arange(0, args.rnn_state_size_lstm_concat, 1)

			time = np.arange(0, len(cell_state_list), 1)
			R= np.array(cell_state_list)
			ax.clear()
			X, Y = np.meshgrid(x_state,time)
			surf = ax.plot_surface(X, Y, R,cmap=cm.coolwarm)
			ax.view_init(elev=90,azim=90)
			ax.set_xlabel("Time [s]")
			ax.set_xlabel("Features")
			fig.colorbar(surf)
			save_img_to_file = model_path + "/figs"
			if not os.path.exists(save_img_to_file):
				os.makedirs(save_img_to_file)
			pl.savefig(save_img_to_file+"/cell_state" + str(exp_idx) + '.png')
			pl.close()

			fig = pl.figure()
			ax = pl.axes(projection="3d")
			time = np.arange(0, len(cell_ped_list), 1)
			R= np.array(cell_ped_list)
			ax.clear()
			X, Y = np.meshgrid(x_ped,time)
			surf = ax.plot_surface(X, Y, R,cmap=cm.coolwarm)
			ax.view_init(elev=90,azim=90)
			ax.set_xlabel("Time [s]")
			ax.set_xlabel("Features")
			fig.colorbar(surf)
			pl.savefig(save_img_to_file+"/cell_ped" + str(exp_idx) + '.png')
			pl.close()

			fig = pl.figure()
			ax = pl.axes(projection="3d")
			time = np.arange(0, len(cell_concat_list), 1)
			R= np.array(cell_concat_list)
			ax.clear()
			X, Y = np.meshgrid(x_concat,time)
			surf = ax.plot_surface(X, Y, R,cmap=cm.coolwarm)
			ax.view_init(elev=90,azim=90)
			ax.set_xlabel("Time [s]")
			ax.set_xlabel("Features")
			fig.colorbar(surf)
			pl.savefig(save_img_to_file+"/cell_concat" + str(exp_idx) + '.png')
			pl.close()
		trajectories.append(trajectory)
		all_predictions.append(predictions)
		input_list.append(x_input_series)
		goal_list.append(goal_input_series)
		grid_list.append(grid_input_series)
		ped_grid_list.append(ped_grid_series)
		y_ground_truth_list.append(y_ground_truth_series)
		other_agents_list.append(other_agents)
		# update progress bar

sess.close()

# Save data
if not os.path.exists(args.model_path + '/../results/'):
	os.makedirs(args.model_path + '/../results/')
results= {
	"trajectories": trajectories,
	"predictions": all_predictions
}
scenario = args.scenario.split('/')[-1]
results_file = args.model_path + '/../results/'+scenario + str(args.exp_num) + "_results.mat"
if os.path.exists(results_file):
	sio.savemat(results_file,results)
else:
	print("Directory not found to save results: " + results_file)

if test_args.record:
		recorder = rec(args, data_prep.agent_container.occupancy_grid)
		if ( "real_world" in test_args.scenario) and not test_args.unit_testing:
			print("Real data!!")
			recorder.plot_on_video(input_list, grid_list, ped_grid_list, all_predictions, y_ground_truth_list,
			                       other_agents_list,
			                       trajectories, test_args)
		else:
			recorder.plot_on_image(input_list, grid_list, ped_grid_list, all_predictions, y_ground_truth_list, other_agents_list,
				                       trajectories,test_args)
			recorder.animate_local(input_list, grid_list, ped_grid_list, all_predictions, y_ground_truth_list, other_agents_list,
		                 trajectories,test_args)
		print("Recorder is done!")
else:
	"""Performance tests"""
	print("Average loss: " + str(np.mean(np.asarray(batch_loss))))
	pred_error, pred_error_summary_lstm = compute_trajectory_prediction_mse(args, trajectories, all_predictions)
	pred_fde, pred_error_summary_lstm_fde = compute_trajectory_fde(args, trajectories, all_predictions)
	args.scenario = training_scenario
	write_results_summary(np.mean(pred_error_summary_lstm), np.mean(pred_error_summary_lstm_fde), 0, args, test_args)

import sys
import os

sys.path.append('../')
import numpy as np
import argparse
import pylab as pl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import pickle as pkl
import importlib
# import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import random
import progressbar
import scipy.io as sio
import json
from copy import deepcopy
if sys.version_info[0] < 3:
	sys.path.append('../src/data_utils')
	sys.path.append('../src/models')
	import DataHandler as dh
	import DataHandlerLSTM as dhlstm
	from plot_utils import *
	import Support as sup
	from Performance import *
	from utils import *
	import Recorder as rec
else:
	from src.data_utils import DataHandler as dh
	from src.data_utils import DataHandlerLSTM as dhlstm
	from src.data_utils.plot_utils import *
	from src.data_utils import Support as sup
	from src.data_utils.Performance import *
	from src.data_utils.utils import *
	from src.data_utils.Recorder import Recorder as rec


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
# args.data_path = "../data/2_agents_swap/trajs/"
# args.scenario = "GA3C-CADRL-10-py27"
# args.dataset = args.scenario+'.pkl'
# args.data_path = '../data/'

training_scenario = args.scenario
args.scenario = test_args.scenario
args.real_world_data = test_args.real_world_data
args.dataset = '/' + args.scenario + '.pkl'
data_prep = dhlstm.DataHandlerLSTM(args)
# Only used to create a map from png
map_args = {"file_name": 'map.png',
	            "resolution": 0.1,
	            "map_size": np.array([30., 30.]), }

data_prep.processData(**map_args)
if args.normalize_data:
	data_prep.compute_min_max_values()

# Import model
module = importlib.import_module("src.models."+args.model_name)
globals().update(module.__dict__)

model = NetworkModel(args)

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

	for exp_idx in range(0,np.minimum(test_args.num_test_sequences,len(data_prep.trajectory_set)-1),args.batch_size):
		predictions = []

		batch_x_list, batch_vel_list, batch_pos_list, batch_goal_list, batch_grid_list, \
		pedestrian_grid_list, batch_y_list, batch_pos_final_list, other_agents_pos_list, \
		traj_list = data_prep.getGroupOfTrajectoriesAsBatch(exp_idx,n_trajs=6)  # trajectory_set random.randint(0, len(data_prep.dataset) - 1)

		trajectories.append(traj_list)

		x_input_series = []
		goal_input_series = []
		grid_input_series = []
		y_ground_truth_series = []

		for traj_id in range(len(traj_list)):
			batch_x = batch_x_list[traj_id]
			batch_vel = batch_vel_list[traj_id]
			batch_pos = batch_pos_list[traj_id]
			batch_goal = batch_goal_list[traj_id]
			batch_grid = batch_grid_list[traj_id]
			other_agents_info = pedestrian_grid_list[traj_id]
			batch_pos_final = batch_pos_final_list[traj_id]
			other_agents_pos = other_agents_pos_list[traj_id]
			batch_grid = batch_grid_list[traj_id]
			traj = traj_list[traj_id]

			ped_grid_series = np.zeros([0, args.n_other_agents, args.pedestrian_vector_dim * args.prediction_horizon])
			model.reset_test_cells(np.ones((args.batch_size)))

			samples = []
			for step in range(batch_x.shape[1]):
				# Assemble feed dict for training
				if "future" in args.others_info:
					if step == 0:
						batch_y_pred = deepcopy(batch_vel)
						for pred_step in range(1, args.prediction_horizon):
							batch_y_pred[:, :, pred_step * 2:pred_step * 2 + 2] = batch_vel[:, :, :2]
					else:
						batch_y_pred = deepcopy(batch_vel)
						batch_y_pred[:, :, 2:] = y_model_pred[:, :, 2:]
				dict = {"batch_x": batch_vel,
				        "batch_grid": batch_grid,
				        "batch_ped_grid": other_agents_info,
				        "step": step,
				}
				feed_dict_ = model.feed_test_dic(**dict)

				y_model_pred, output_decoder, cell_outputs_series_state, cell_outputs_series_lstm_ped, cell_concat = model.predict(sess, feed_dict_, True)

				samples.append(y_model_pred[:,0,:])

			predictions.append(samples)

		all_predictions.append(predictions)
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
results_file = args.model_path + '/'+ scenario + "_results.mat"

sio.savemat(results_file,results)

if test_args.record:
		recorder = rec(args, data_prep.agent_container.occupancy_grid)
		if ( "real_world" in test_args.scenario) and not test_args.unit_testing:
			print("Real data!!")
			recorder.plot_on_video(input_list, grid_list, all_predictions, y_ground_truth_list,
			                       other_agents_list,
			                       trajectories, test_args)
		else:
			#recorder.plot_on_image(input_list, grid_list, all_predictions, y_ground_truth_list, other_agents_list,
			#	                       trajectories,test_args)
			#recorder.animate_local(input_list, grid_list, ped_grid_list, all_predictions, y_ground_truth_list, other_agents_list,
		  #               trajectories,test_args)
			recorder.animate_group_of_agents(trajectories, all_predictions,test_args)

		print("Recorder is done!")
else:
	"""Performance tests"""
	print("Average loss: " + str(np.mean(np.asarray(batch_loss))))
	pred_error, pred_error_summary_lstm = compute_trajectory_prediction_mse(args, trajectories, all_predictions)
	pred_fde, pred_error_summary_lstm_fde = compute_trajectory_fde(args, trajectories, all_predictions)
	args.scenario = training_scenario
	write_results_summary(np.mean(pred_error_summary_lstm), np.mean(pred_error_summary_lstm_fde), 0, args, test_args)

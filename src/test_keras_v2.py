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

if sys.version_info[0] < 3:
	sys.path.append(sys.path[0] + '/data_utils') # sys.path.append('../src/data_utils')
	sys.path.append(sys.path[0] + '/models') # sys.path.append('../src/models')
	import DataHandler_Keras_v2 as dh
	import DataHandlerLSTM as dhlstm
	from plot_utils import *
	import Support as sup
	from Performance import *
	from utils import *
	import Recorder_Keras as rec
else:
    from src.data_utils import DataHandler_Keras_v2 as dh
    from src.data_utils import DataHandlerLSTM as dhlstm
    from src.data_utils.plot_utils import *
    from src.data_utils import Support as sup
    from src.data_utils.Performance import *
    from src.data_utils.utils import *
    from src.data_utils.Recorder_Keras import Recorder as rec


# Define default args
model_name = "modelKerasRNN_arbitraryAgents"
exp_num = 205

root_folder = os.path.dirname(sys.path[0])
data_path = root_folder + '/data/'

scenario = "2_agents/trajs/GA3C-CADRL-10"
# scenario = "2_agents_swap/trajs/GA3C-CADRL-10-py27"
# scenario = "2_agents_random/trajs/GA3C-CADRL-10"

defaults = {
    "model_name": model_name,
    "exp_num": exp_num,
    "scenario":scenario,
    "data_path": data_path
}

# Model directories
test_args = parse_args(defaults, "Test")

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

if "GA3C" in args.scenario:
	data_prep = dh.DataHandler(args)
	# Collect data online
	# if os.path.isfile(args.data_path + args.dataset):
	# 	data_prep.load_data()
	# else:
	# 	data_prep.start_node()
	assert os.path.isfile(args.data_path + args.dataset), "File " + args.data_path + args.dataset + " does not exist"
	data_prep.load_data()
	dataset_size = len(data_prep.trajectory_set) - 1
else:
	data_prep = dhlstm.DataHandlerLSTM(args)
	# Only used to create a map from png
	map_args = {"file_name": 'map.png',
	            "resolution": 0.1,
	            "map_size": np.array([20., 7.]), }
	data_pickle = args.data_path + args.scenario + "/data" + str(args.prediction_horizon) + ".pickle"
	data_prep.processData(data_pickle, **map_args)
	dataset_size = len(data_prep.trajectory_set) - 1
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
	args.normalize_data = False

# Import model
model = model_selector(args)
model.compile(loss=model.loss_object, optimizer=model.optimizer)

# TODO: DataHandlerLSTM for Keras model has not been implemented yet
# if test_args.unit_testing:
# 	data_handler = dhlstm.DataHandlerLSTM(args)
# 	data_handler.unit_test_data_()


# We need to first call the model in order to be able to load the weights
batch_example = data_prep.getTrajectoryAsBatch(0)
model.call(batch_example['input'])
model.load_weights(args.model_path + '/model_ckpt.h5')

if model.stateful: # Set model back to stateless for prediction
    for i in range(len(model.layers)):
        model.layers[i].stateful = False

# TODO: Convnet warmstart
# try:
# 		model.warmstart_convnet(args, sess)
# except:
	# print("")

all_predictions = []
all_trajectories = []

for exp_idx in range(test_args.num_test_sequences):
    batch = data_prep.getPaddedTrajectoryAsBatch(exp_idx)
    prediction = model.predict(batch['input'])
    all_predictions.append( prediction )
    all_trajectories.append( batch['trajectory'] )

rec(all_trajectories, all_predictions, args, save = test_args.record, display = False)

print("Test script for model " + test_args.model_name + "/" + str(test_args.exp_num) + " finished")
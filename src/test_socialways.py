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
import progressbar
import torch
import numpy as np
import torch.nn as nn
import torch.optim as opt
from tqdm import tqdm, trange
from itertools import chain
from torch.autograd import Variable

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
  from socialways import *
else:
  from src.data_utils import DataHandler as dh
  from src.data_utils import DataHandlerLSTM as dhlstm
  from src.data_utils.plot_utils import *
  from src.data_utils import Support as sup
  from src.data_utils.Performance import *
  from src.data_utils.utils import *
  from src.data_utils.Recorder import Recorder as rec
  from src.external.socialways import *


# Model directories
def parse_args():
  parser = argparse.ArgumentParser(description='LSTM model training')

  parser.add_argument('--model_name',
                      help='Path to directory that comprises the model (default="model_name").',
                      type=str, default="VGDNN_simple")
  parser.add_argument('--num_test_sequences', help='Number of test sequences', type=int, default=10)
  parser.add_argument('--exp_num', help='Experiment number', type=int, default=50)
  parser.add_argument('--n_samples', help='Number of samples', type=int, default=10)
  parser.add_argument('--scenario', help='Scenario of the dataset (default="").',
                      type=str, default="zara_01")
  parser.add_argument('--record', help='Is grid rotated? (default=True).', type=sup.str2bool,
                      default=True)
  parser.add_argument('--save_figs', help='Save figures?', type=sup.str2bool,
                      default=False)
  parser.add_argument('--noise_cell_state', help='Adding noise to cell state of the agent', type=float,
                      default=0)
  parser.add_argument('--noise_cell_grid', help='Adding noise to cell state of the grid', type=float,
                      default=0)
  parser.add_argument('--real_world_data', help='real_world_data', type=sup.str2bool,
                      default=False)
  parser.add_argument('--update_state', help='update_state', type=sup.str2bool,
                      default=True)
  parser.add_argument('--gpu', help='Enable GPU training.', type=sup.str2bool,
                      default=True)
  args = parser.parse_args()

  return args

test_args = parse_args()

if test_args.gpu:
	import tensorflow as tf
else:
	os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
	import tensorflow as tf

cwd = os.getcwd()

model_path = os.path.normpath(cwd+'/../') +'/trained_models/'+ test_args.model_name + "/" + str(test_args.exp_num)

print("Loading data from: '{}'".format(model_path))
file = open(model_path + '/model_parameters.pkl', 'rb')
if sys.version_info[0] < 3:
  model_parameters = pkl.load(file)#,encoding='latin1')
else:
  model_parameters = pkl.load(file , encoding='latin1')
file.close()
args = model_parameters["args"]

# change some args because we are doing inference
args.truncated_backprop_length = 1
args.batch_size = 1
#args.data_path = "../data/2_agents_swap/trajs/"
#args.scenario = "GA3C-CADRL-10-py27"
#args.dataset = args.scenario+'.pkl'
#args.data_path = '../data/'
training_scenario = args.scenario
args.scenario = test_args.scenario
args.real_world_data = test_args.real_world_data

if "GA3C" in args.scenario:
	data_prep = dh.DataHandler(args)
	# Collect data online
	if os.path.isfile(args.data_path + args.dataset):
		data_prep.load_data()
	else:
		data_prep.start_node()
	dataset_size = len(data_prep.dataset) -1
else:
	data_prep = dhlstm.DataHandlerLSTM(args)
	# Only used to create a map from png
	map_args = {"file_name": 'map.png',
	            "resolution": 0.1,
	            "map_size": np.array([20., 7.]), }
	data_pickle = args.data_path + args.scenario + "/data" + str(args.prediction_horizon) + ".pickle"
	data_prep.processData(data_pickle,**map_args)
	dataset_size = len(data_prep.trajectory_set) -1

	args.normalize_data = False

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
		from modelRNN_v2 import *
	else:
		from src.models.modelRNN_v2 import *
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
elif "simple" in args.model_name:
	if sys.version_info[0] < 3:
		from VGDNN_simple import *
	else:
		from src.models.VGDNN_simple import *
else:
	if sys.version_info[0] < 3:
		from VGDNN import *
	else:
		from src.models.VGDNN import *

model = NetworkModel(args)

"""Social Ways Model"""
model_name = 'socialWays'
model_file = '../trained_models/' + model_name + '-' + test_args.scenario + '.pt'

if "eth" in test_args.scenario:
	model_file = '../trained_models/' + model_name + '-eth.pt'
if "hotel" in test_args.scenario:
	model_file = '../trained_models/' + model_name + '-hotel.pt'

encoder = EncoderLstm(hidden_size, n_lstm_layers).cuda()
feature_embedder = EmbedSocialFeatures(num_social_features, social_feature_size).cuda()
attention = AttentionPooling(hidden_size, social_feature_size).cuda()
#use_social = False
# Decoder
decoder = DecoderFC(hidden_size + social_feature_size + noise_len).cuda()
# decoder = DecoderLstm(social_feature_size + VEL_VEC_LEN + noise_len, traj_code_len).cuda()

# The Generator parameters and their optimizer
predictor_params = chain(attention.parameters(), feature_embedder.parameters(),
                         encoder.parameters(), decoder.parameters())
predictor_optimizer = opt.Adam(predictor_params, lr=lr_g, betas=(0.9, 0.999))

# The Discriminator parameters and their optimizer
D = Discriminator(n_next, hidden_size, n_latent_codes).cuda()
D_optimizer = opt.Adam(D.parameters(), lr=lr_d, betas=(0.9, 0.999))

data_prep.calc_scale()

print('hidden dim = %d | lr(G) =  %.5f | lr(D) =  %.5f' % (hidden_size, lr_g, lr_d))

if os.path.isfile(model_file):
    print('Loading model from ' + model_file)
    checkpoint = torch.load(model_file)
    start_epoch = checkpoint['epoch'] + 1

    attention.load_state_dict(checkpoint['attentioner_dict'])
    feature_embedder.load_state_dict(checkpoint['feature_embedder_dict'])
    encoder.load_state_dict(checkpoint['encoder_dict'])
    decoder.load_state_dict(checkpoint['decoder_dict'])
    predictor_optimizer.load_state_dict(checkpoint['pred_optimizer'])

    D.load_state_dict(checkpoint['D_dict'])
    D_optimizer.load_state_dict(checkpoint['D_optimizer'])

# Lists for logging of the input / output data of the model
input_list = []
grid_list = []
ped_grid_list = []
y_ground_truth_list = []
y_pred_list = []  # uses ground truth as input at every step
other_agents_list = []
all_predictions = []
all_social_prediction = []
trajectories = []
batch_y = []
batch_loss = []

p_bar = progressbar.ProgressBar(widgets=[progressbar.Percentage(), progressbar.Bar()],
                                maxval=test_args.num_test_sequences).start()

config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )

def predict(obsv_p, noise, n_next, sub_batches=[]):
    # Batch size
    bs = obsv_p.shape[0]
    # Adds the velocity component to the observations.
    # This makes of obsv_4d a batch_sizexTx4 tensor
    obsv_4d = get_traj_4d(obsv_p, [])
    # Initial values for the hidden and cell states (zero)
    lstm_h_c = (torch.zeros(n_lstm_layers, bs, encoder.hidden_size).cuda(),
                torch.zeros(n_lstm_layers, bs, encoder.hidden_size).cuda())
    encoder.init_lstm(lstm_h_c[0], lstm_h_c[1])
    # Apply the encoder to the observed sequence
    # obsv_4d: batch_sizexTx4 tensor
    encoder(obsv_4d)
    if len(sub_batches) == 0:
        sub_batches = [[0, obsv_p.size(0)]]

    if use_social:
        features = SocialFeatures(obsv_4d, sub_batches)
        emb_features = feature_embedder(features, sub_batches)
        weighted_features = attention(emb_features, encoder.lstm_h[0].squeeze(), sub_batches)
    else:
        weighted_features = torch.zeros_like(encoder.lstm_h[0].squeeze())

    pred_4ds = []
    last_obsv = obsv_4d[:, -1]
    # For all the steps to predict, applies a step of the decoder
    for ii in range(n_next):
        # Takes the current output of the encoder to feed the decoder
        # Gets the ouputs as a displacement/velocity
        new_v = decoder(encoder.lstm_h[0].view(bs, -1), weighted_features.view(bs, -1), noise).view(bs, 2)
        # Deduces the predicted position
        new_p = new_v + last_obsv[:, :2]
        # The last prediction done will be new_p,new_v
        last_obsv = torch.cat([new_p, new_v], dim=1)
        # Keeps all the predictions
        pred_4ds.append(last_obsv)
        # Applies LSTM encoding to the last prediction
        # pred_4ds[-1]: batch_sizex4 tensor
        encoder(pred_4ds[-1])

    return torch.stack(pred_4ds, 1)

with tf.Session(config=config) as sess:

	model.warmstart_model(args, sess)
	model.warmstart_convnet(args, sess)

	for exp_idx in range(0, min(dataset_size,test_args.num_test_sequences)):
		predictions = []
		social_predictions = []
		batch_x, batch_vel,batch_pos, batch_grid, other_agents_info, batch_target, other_agents_pos, traj = data_prep.getTrajectoryAsBatch(
			exp_idx) # trajectory_set random.randint(0, len(data_prep.dataset) - 1)

		if args.agents_on_grid:
			batch_grid_with_agents = batch_grid.copy()
			data_prep.add_other_agents_to_grid(batch_grid_with_agents, batch_x, [other_agents_pos], args)

		trajectories.append(traj)
		x_input_series = np.zeros([0, (args.prev_horizon + 1) * args.input_dim])
		grid_input_series = np.zeros([0, int(args.submap_width/args.submap_resolution), int(args.submap_height/args.submap_resolution)])
		ped_grid_series = np.zeros([0, args.pedestrian_vector_dim])
		y_ground_truth_series = np.zeros([0, args.prediction_horizon * 2])
		y_pred_series = np.zeros([0, args.n_mixtures*args.prediction_horizon * args.output_pred_state_dim])

		batch_y.append(batch_target)
		model.reset_test_cells(np.ones((args.batch_size)))

		for step in range(batch_x.shape[1]):
			samples = []
			social_samples = []
			# Assemble feed dict for training
			if args.input_state_dim < 4:
				x = np.expand_dims(batch_vel[:, step, :], axis=0)
			else:
				x = np.expand_dims(batch_x[:, step, :], axis=0)

			agents = np.expand_dims(other_agents_info[:, step, :], axis=0)
			grid = np.expand_dims(batch_grid[:, step, :, :], axis=0)
			y = np.expand_dims(batch_target[:, step, :], axis=0)
			pos = np.expand_dims(batch_pos[:, step, :], axis=0)
			try:
				feed_dict_ = model.feed_test_dic(x, grid,agents,y,pos,0)
			except:
				feed_dict_ = model.feed_test_dic(x, grid, agents, y, 0)

			# Assemble Pytorch model feed dict for training
			positions = np.zeros((1,args.prev_horizon+1,2))
			velocities = np.zeros((1,args.prev_horizon+1,2))
			for j in range(int(batch_x.shape[2]/4)):
				positions[0,args.prev_horizon-j,:] = [batch_x[:, step, j*4],batch_x[0, step, j*4+1]]

			positions_norm = data_prep.normalize_pos(positions,inPlace=False)
			obsv = torch.FloatTensor(positions_norm).cuda()
			for sample_id in range(3):
				noise = torch.FloatTensor(torch.rand(1, noise_len)).cuda()
				pred_hat_4d = predict(obsv, noise, n_next)

				social_predicted_positions = pred_hat_4d.data.cpu().numpy()[0,:,:2]
				social_predicted_velocities = pred_hat_4d.data.cpu().numpy()[0, :, 2:]
				social_predicted_positions_normalized = data_prep.denormalize(social_predicted_positions,inPlace=False)
				social_predicted_velocities_normalized = data_prep.denormalize(social_predicted_velocities,shift=False,inPlace=False)

				social_samples.append(social_predicted_velocities_normalized)

			social_predictions.append(social_samples)

			# Append to logging series
			x_input_series = np.append(x_input_series,batch_x[:, step, :], axis=0)
			grid_input_series = np.append(grid_input_series, batch_grid[:, step, :,:], axis=0)
			ped_grid_series = np.append(ped_grid_series, other_agents_info[:, step, :], axis=0)
			y_ground_truth_series = np.append(y_ground_truth_series, batch_target[:, step, :], axis=0)

			y_model_pred, loss, output_decoder = model.predict(sess, feed_dict_, True)
			samples.append(y_model_pred[0])

			for sample_id in range(test_args.n_samples-1):
				try:
					feed_dict_ = model.feed_test_dic(x, grid, agents, y, pos, 0)
				except:
					feed_dict_ = model.feed_test_dic(x, grid, agents, y, 0)

				y_model_pred,loss, output_decoder = model.predict(sess, feed_dict_, test_args.update_state)
				samples.append(y_model_pred[0])

			batch_loss.append(loss)

			predictions.append(samples)

		all_predictions.append(predictions)
		all_social_prediction.append(social_predictions)
		input_list.append(x_input_series)
		grid_list.append(grid_input_series)
		ped_grid_list.append(ped_grid_series)
		y_ground_truth_list.append(y_ground_truth_series)
		other_agents_list.append(other_agents_pos)
		# update progress bar
		p_bar.update(exp_idx)


sess.close()
#matplot_dataset(trajectories,args)
#plot_local_scenario_vel_OpenCV(trajectories, batch_y,all_predictions, args)
#plot_scenario_vel_OpenCV_simdata(trajectories, batch_y,all_predictions, args)

if test_args.record:
  recorder = rec(args,data_prep.agent_container.occupancy_grid)
  if test_args.real_world_data:
    print("Real data!!")
    recorder.plot_on_image(input_list, grid_list, ped_grid_list, all_predictions, y_ground_truth_list, other_agents_list, trajectories,args.rotated_grid, test_args.n_samples,all_social_prediction)
  else:
    recorder.animate(input_list, grid_list, ped_grid_list, all_predictions, y_ground_truth_list, other_agents_list, args.rotated_grid, test_args.n_samples, test_args.save_figs)
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

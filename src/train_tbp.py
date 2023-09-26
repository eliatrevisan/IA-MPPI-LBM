import sys
import os
sys.path.append('../')
import numpy as np
import argparse
import tensorflow as tf
# import tensorflow_probability as tfp
import matplotlib.pyplot as plt
if sys.version_info[0] < 3:
  sys.path.append('../src/data_utils')
  sys.path.append('../src/models')
  import DataHandler as dh
  from seqRNNModel import *
else:
  from src.data_utils import DataHandler as dh
  from src.models.seqRNNModel import *
  from src.data_utils.plot_utils import *

tfk = tf.keras

model_name = "seq_to_seq"
input_dim = 8
latent_dim = 256
output_dim = 2
batch_size = 16
learning_rate_init = 0.01
grads_clip = 10.0
n_epochs = 100
stateful = False
prediction_horizon = 10
prev_horizon = 10
truncated_backprop_length = 10
regularization_weight = 0.0001
dropout = 0.2

def parse_args():
	parser = argparse.ArgumentParser(description='LSTM model training')

	parser.add_argument('--model_name',
	                    help='Path to directory that comprises the model (default="model_name").',
	                    type=str, default=model_name)
	parser.add_argument('--data_path', help='Data directory', type=str,
	                    default='../data')
	parser.add_argument('--input_dim', help='Input state dimension (default=).', type=int,
	                    default=input_dim)
	parser.add_argument('--latent_dim', help='Latent dimension.', type=int,
	                    default=latent_dim)
	parser.add_argument('--prev_horizon', help='Previous seq length.', type=int,
	                    default=prev_horizon)
	parser.add_argument('--prediction_horizon', help='Prediction Horizon.', type=int,
	                    default=prediction_horizon)
	parser.add_argument('--output_dim', help='Output dimension.', type=int,
	                    default=output_dim)
	parser.add_argument('--batch_size', help='Batch size.', type=int,
	                    default=batch_size)
	parser.add_argument('--learning_rate_init', help='Learning rate.', type=float,
	                    default=learning_rate_init)
	parser.add_argument('--grads_clip', help='Grad clip.', type=float,
	                    default=grads_clip)
	parser.add_argument('--n_epochs', help='Number of epochs.', type=float,
	                    default=n_epochs)
	parser.add_argument('--stateful', help='Stateful.', type=bool,
	                    default=stateful)
	parser.add_argument('--truncated_backprop_length', help='Backpropagation length during training (default=5).',
	                    type=int, default=truncated_backprop_length)
	parser.add_argument('--regularization_weight', help='Weight scaling of regularizer (default=0.01).', type=float,
	                    default=regularization_weight)
	parser.add_argument('--dropout', help='Dropout probability (default=0.005).', type=float,
	                    default=dropout)
	args = parser.parse_args()

	return args


args = parse_args()

data_prep = dh.DataHandler()
# Collect data online
if os.path.isfile(data_prep.args.save_path):
	data_prep.load_data()
else:
	data_prep.start_node()

batch_enc, batch_dec, batch_dec_target, new_epoch = data_prep.getBatchTBP()

model = RNNModel(args)

model.model.compile(optimizer=tfk.optimizers.Adam(lr=args.learning_rate_init,clipnorm=args.grads_clip),
                             loss='mean_squared_error')


history = model.model.fit([batch_enc,batch_dec],batch_dec_target,
                epochs=n_epochs,
                batch_size=16,
								#validation_split=0.2,
                shuffle=True)

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.title('Model loss during training')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper left')
plt.show()

batch_enc_seq, batch_dec_seq, traj = data_prep.getTrajectoryAsBatchTBP()

predictions = model.decode_sequence([batch_enc_seq,batch_dec_seq])

#plot_scenario(batch_enc_seq,batch_dec_seq,traj,predictions,data_prep.occupancy_grid,args)
plot_scenario_vel(batch_enc_seq,batch_dec_seq,traj,predictions,data_prep.occupancy_grid,args)
print("test")


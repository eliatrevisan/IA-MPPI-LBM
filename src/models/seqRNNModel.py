import tensorflow as tf
#import tensorflow_probability as tfp
import numpy as np
import os
import sys

sys.path.append('./autoencoder_grid')
tfk = tf.keras
tfkl = tf.keras.layers
#tfpl = tfp.layers
#tfd = tfp.distributions


class RNNModel():

	def __init__(self, args, batch_size=None):
		# Input- / Output dimensions
		if args.input_dim is not None:
			self.input_dim = args.input_dim
		if args.latent_dim is not None:
			self.latent_dim = args.latent_dim
		if args.output_dim is not None:
			self.output_dim = args.output_dim
		if args.batch_size is not None:
			self.batch_size = args.batch_size
		self.args = args

		# Define an input sequence and process it.
		self.encoder_inputs = tfkl.Input(shape=(None,self.input_dim*self.args.prev_horizon))

		# Encoder state LSTM
		self.enc_state_lstm = tfkl.LSTM(self.latent_dim,
		                                  input_shape=[None, self.input_dim],
		                                  dropout=self.args.dropout, name='enc_lstm', return_sequences=True,
		                                  return_state=True, stateful=args.stateful,
		                                  kernel_regularizer=tfk.regularizers.l2(l=self.args.regularization_weight))

		self.encoder_outputs, self.state_h, self.state_c = self.enc_state_lstm(self.encoder_inputs)

		# We discard `encoder_outputs` and only keep the states.
		self.enc_states = [self.state_h, self.state_c]

		# Set up the decoder, using `encoder_states` as initial state.
		self.dec_inputs = tfkl.Input(shape=(None, self.args.output_dim))

		self.decoder_lstm = tfkl.LSTM(self.latent_dim,
		                             input_shape=[None, self.latent_dim],
		                             dropout=self.args.dropout, name='dec_lstm',
		                             return_sequences=True, return_state=True, stateful=args.stateful,
		                             kernel_regularizer=tfk.regularizers.l2(l=self.args.regularization_weight))

		self.decoder_o, _, _ = self.decoder_lstm(self.dec_inputs, initial_state=self.enc_states)

		self.fcl = tfkl.TimeDistributed(tfkl.Dense(self.latent_dim, activation='relu', name='dec_fc',
		                                           kernel_regularizer=tfk.regularizers.l2(l=self.args.regularization_weight)))
		self.fco = tfkl.TimeDistributed(tfkl.Dense(self.output_dim*self.args.prediction_horizon, activation='linear', name='out_fc',
		                                           kernel_regularizer=tfk.regularizers.l2(l=self.args.regularization_weight)))

		self.dec_outputs = self.fcl(self.decoder_o)

		self.dec_outputs = self.fco(self.dec_outputs)

		# Define the model that will turn
		# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
		self.model = tfk.Model([self.encoder_inputs, self.dec_inputs], self.dec_outputs)

		self.encoder_state_input_h = tfkl.Input(shape=(self.latent_dim,))
		self.encoder_state_input_c = tfkl.Input(shape=(self.latent_dim,))
		self.encoder_states_inputs = [self.encoder_state_input_h, self.encoder_state_input_c]

		self.encoder_model = tfk.Model([self.encoder_inputs] + self.encoder_states_inputs, self.enc_states)

		self.decoder_state_input_h = tfkl.Input(shape=(self.latent_dim,))
		self.decoder_state_input_c = tfkl.Input(shape=(self.latent_dim,))
		self.decoder_states_inputs = [self.decoder_state_input_h, self.decoder_state_input_c]

		self.decoder_o, self.dec_state_h, self.dec_state_c = self.decoder_lstm(self.dec_inputs,
		                                                                       initial_state=self.decoder_states_inputs)

		self.dec_states = [self.dec_state_h, self.dec_state_c]

		self.dec_out = self.fcl(self.decoder_o)

		self.dec_out = self.fco(self.dec_out)

		self.decoder_model = tfk.Model([self.dec_inputs] + self.decoder_states_inputs, [self.dec_out] + self.dec_states)

	def decode_sequence(self,input_seq):
		# Populate the first character of target sequence with the start character.
		target_seq = np.zeros((1, 1, self.args.output_dim))
		predictions = []
		states_value = [np.zeros((1, self.args.latent_dim)), np.zeros((1, self.args.latent_dim))]
		for i in range(len(input_seq[0])):
			# Encode the input as state vectors.
			in_enc = [input_seq[0][i]]

			target_seq[0, 0, :] = input_seq[1][i]
			states_value = self.encoder_model.predict([in_enc]+states_value,batch_size=1)

			decoded_sentence = np.zeros((1, self.args.output_dim*self.args.prediction_horizon))


			output_tokens, h, c = self.decoder_model.predict(
					[target_seq] + states_value)

			# Sample a token
			sampled_output = output_tokens[0, -1, :]

			decoded_sentence[0, :] = sampled_output

			# Update states
			states_value = [h, c]

			predictions.append(decoded_sentence.copy())

		return predictions
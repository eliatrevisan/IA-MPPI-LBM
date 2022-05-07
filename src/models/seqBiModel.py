import tensorflow as tf
# import tensorflow_probability as tfp
import numpy as np
import os
import sys

tfk = tf.keras
tfkl = tf.keras.layers
# tfpl = tfp.layers
# tfd = tfp.distributions

class SeqToSeqModel2():#tfk.Model

	def __init__(self, args):
		#used for subclassing
		#super(SeqToSeqModel2, self).__init__()
		#Network parameters
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
		self.encoder_inputs = tfkl.Input(shape=(None, self.input_dim))
		#During training we should use the previous hidden_states right ?
		self.encoder_lstm = tfkl.Bidirectional(tfkl.LSTM(self.latent_dim, return_state=True,stateful=args.stateful))

		self.encoder_outputs, self.state_h_f, self.state_c_f, self.state_h_b, self.state_c_b = self.encoder_lstm(self.encoder_inputs)

		# We discard `encoder_outputs` and only keep the states.
		#self.enc_states_f = [self.state_h_f, self.state_c_f]
		#self.enc_states_b = [self.state_h_b, self.state_c_b]

		self.enc_state_h = tfkl.Concatenate()([self.state_h_f, self.state_h_b])
		self.enc_state_c = tfkl.Concatenate()([self.state_c_f, self.state_c_b])

		self.enc_states = [self.enc_state_h, self.enc_state_c]

		# Set up the decoder, using `encoder_states` as initial state.
		self.dec_inputs = tfkl.Input(shape=(None, self.output_dim))

		self.decoder_lstm = tfkl.LSTM(self.latent_dim * 2, return_sequences=True, return_state=True)

		self.decoder_o, _, _ = self.decoder_lstm(self.dec_inputs,initial_state=self.enc_states)

		self.decoder_fcl = tfkl.Dense(self.latent_dim,activation=tfk.activations.relu)

		self.decoder_linear = tfkl.Dense(self.output_dim)

		self.dec_outputs = self.decoder_fcl(self.decoder_o)

		self.dec_outputs = self.decoder_linear(self.dec_outputs)

		# Define the model that will turn
		# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
		self.model = tfk.Model([self.encoder_inputs, self.dec_inputs], self.dec_outputs)

		self.encoder_state_input_h = tfkl.Input(shape=(self.latent_dim,))
		self.encoder_state_input_c = tfkl.Input(shape=(self.latent_dim,))
		self.encoder_states_inputs = [self.encoder_state_input_h, self.encoder_state_input_c]

		self.encoder_model = tfk.Model([self.encoder_inputs], self.enc_states)

		self.decoder_state_input_h = tfkl.Input(shape=(self.latent_dim * 2,))
		self.decoder_state_input_c = tfkl.Input(shape=(self.latent_dim * 2,))
		self.decoder_states_inputs = [self.decoder_state_input_h, self.decoder_state_input_c]

		self.decoder_o, self.dec_state_h, self.dec_state_c = self.decoder_lstm(self.dec_inputs,
		                                                                       initial_state=self.decoder_states_inputs)

		self.dec_states = [self.dec_state_h, self.dec_state_c]

		self.dec_out = self.decoder_fcl(self.decoder_o)

		self.dec_out = self.decoder_linear(self.dec_out)

		self.decoder_model = tfk.Model([self.dec_inputs] + self.decoder_states_inputs,[self.dec_out]+ self.dec_states)

	def get_encoder_zero_initial_state(self, inputs):
		return [tf.zeros((self.batch_size, self.latent_dim)), tf.zeros((self.batch_size, self.latent_dim))]

	def get_initial_state(self, inputs):
		return self.initial_state

	"""Used for subclassing
	def __call__(self, inputs, training,states=None):
		if states is None:
			self.encoder_lstm.get_initial_state = self.get_encoder_zero_initial_state
		else:
			self.initial_state = states
			self.encoder_lstm.get_initial_state = self.get_initial_state

		self.encoder_outputs, self.state_h, self.state_c = self.encoder_lstm(inputs[0])

		self.enc_states = [self.state_h, self.state_c]

		self.decoder_outputs, _, _ = self.decoder_lstm(inputs[1], initial_state=self.enc_states)

		self.decoder_dense_o = self.decoder_dense(self.decoder_outputs)

		return self.decoder_dense_o
	"""
	def decode_sequence(self,input_seq):
		# Populate the first character of target sequence with the start character.
		target_seq = np.zeros((1, 1, self.args.output_dim))
		predictions = []
		#states_value = [np.zeros((1, self.args.latent_dim )), np.zeros((1, self.args.latent_dim ))]
		for i in range(len(input_seq[0])):
			# Encode the input as state vectors.
			in_enc = [input_seq[0][i]]
			in_dec = [input_seq[1][i]]
			target_seq[0, 0, :] = in_enc[0][-1, 6:8]
			states_value = self.encoder_model.predict([in_enc],batch_size=1)

			decoded_sentence = np.zeros((self.args.prediction_horizon, self.args.output_dim))
			step = 0
			stop_condition = False
			while not stop_condition:
				# Exit condition: either hit max length
				# or find stop character.
				if step >= self.args.prediction_horizon - 1:
					stop_condition = True

				output_tokens, h, c = self.decoder_model.predict(
					[target_seq] + states_value)

				# Sample a token
				sampled_output = output_tokens[0, -1, :]

				decoded_sentence[step, :] = sampled_output

				# Update the target sequence (of length 1).
				target_seq = output_tokens

				# Update states
				states_value = [h, c]

				step += 1
			predictions.append(decoded_sentence.copy())

		return predictions
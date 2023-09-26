import tensorflow as tf
# import tensorflow_probability as tfp
import numpy as np
import os


class NetworkModel():

	def __init__(self, args, batch_size=None, is_training=True):

		# Input- / Output dimensions
		self.input_dim = args.input_dim  # [x, y, vx, vy]
		self.input_state_dim = args.input_state_dim  # [vx, vy]
		self.output_state_dim = args.output_dim
		self.output_pred_state_dim = args.output_pred_state_dim
		self.truncated_backprop_length = args.truncated_backprop_length
		self.n_mixtures = args.n_mixtures
		self.output_sequence_length = args.prediction_horizon
		self.output_placeholder_dim = self.output_sequence_length * self.output_state_dim
		self.output_dim = self.output_sequence_length * self.output_pred_state_dim
		self.pedestrian_vector_dim = args.pedestrian_vector_dim

		# Network attributes
		self.rnn_state_size = args.rnn_state_size
		self.rnn_state_size_lstm_grid = args.rnn_state_size_lstm_grid
		self.rnn_state_size_lstm_ped = args.rnn_state_size_lstm_ped
		self.rnn_state_size_lstm_concat = args.rnn_state_size_lstm_concat
		self.fc_hidden_unit_size = args.fc_hidden_unit_size

		# Training parameters
		self.lambda_ = args.regularization_weight
		self.batch_size = None
		self.is_training = is_training
		self.grads_clip = args.grads_clip
		self.regularization_weight = args.regularization_weight
		self.keep_prob = args.keep_prob
		# Specify placeholders
		self.input_state_placeholder = tf.placeholder(dtype=tf.float32,
		                                              shape=[self.batch_size, self.truncated_backprop_length,
		                                                     self.input_state_dim], name='input_state')
		self.input_grid_placeholder = tf.placeholder(dtype=tf.float32,
		                                             shape=[self.batch_size, self.truncated_backprop_length,
		                                                    int(args.submap_width/args.submap_resolution),
		                                                    int(args.submap_height/args.submap_resolution)], name='input_grid')
		self.input_ped_grid_placeholder = tf.placeholder(dtype=tf.float32,
		                                                 shape=[self.batch_size, self.truncated_backprop_length,
		                                                        self.pedestrian_vector_dim], name='ped_grid')
		self.output_placeholder = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.truncated_backprop_length,
		                                                                  self.output_placeholder_dim], name='output')

		self.step = tf.placeholder(dtype=tf.float32,
		                           shape=[], name='global_step')

		# Initialize hidden states with zeros
		self.cell_state_current = np.zeros([args.batch_size, args.rnn_state_size])
		self.hidden_state_current = np.zeros([args.batch_size, args.rnn_state_size])
		self.cell_state_current_lstm_grid = np.zeros([args.batch_size, args.rnn_state_size_lstm_grid])
		self.hidden_state_current_lstm_grid = np.zeros([args.batch_size, args.rnn_state_size_lstm_grid])
		self.cell_state_current_lstm_ped = np.zeros([args.batch_size, args.rnn_state_size_lstm_ped])
		self.hidden_state_current_lstm_ped = np.zeros([args.batch_size, args.rnn_state_size_lstm_ped])
		self.cell_state_current_lstm_concat = np.zeros([args.batch_size, args.rnn_state_size_lstm_concat])
		self.hidden_state_current_lstm_concat = np.zeros([args.batch_size, args.rnn_state_size_lstm_concat])

		# Pedestrian state LSTM
		self.cell_state = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.rnn_state_size],
		                                 name='cell_state')  # internal state of the cell (before output gate)
		self.hidden_state = tf.placeholder(dtype=tf.float32, shape=[None, self.rnn_state_size],
		                                   name='hidden_state')  # output of the cell (after output gate)
		self.init_state_tuple = tf.contrib.rnn.LSTMStateTuple(self.cell_state, self.hidden_state)

		# Static occupancy grid LSTM
		self.cell_state_lstm_grid = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.rnn_state_size_lstm_grid],
		                                           name='cell_state_lstm_grid')  # internal state of the cell (before output gate)
		self.hidden_state_lstm_grid = tf.placeholder(dtype=tf.float32, shape=[None, self.rnn_state_size_lstm_grid],
		                                             name='hidden_state_lstm_grid')  # output of the cell (after output gate)
		self.init_state_tuple_lstm_grid = tf.contrib.rnn.LSTMStateTuple(self.cell_state_lstm_grid,
		                                                                self.hidden_state_lstm_grid)

		# Pedestrian LSTM
		self.cell_state_lstm_ped = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.rnn_state_size_lstm_ped],
		                                          name='cell_state_lstm_ped')  # internal state of the cell (before output gate)
		self.hidden_state_lstm_ped = tf.placeholder(dtype=tf.float32, shape=[None, self.rnn_state_size_lstm_ped],
		                                            name='hidden_state_lstm_ped')  # output of the cell (after output gate)
		self.init_state_tuple_lstm_ped = tf.contrib.rnn.LSTMStateTuple(self.cell_state_lstm_ped, self.hidden_state_lstm_ped)

		# Concatenation LSTM (state, grid, pedestrians)
		self.cell_state_lstm_concat = tf.placeholder(dtype=tf.float32,
		                                             shape=[self.batch_size, self.rnn_state_size_lstm_concat],
		                                             name='cell_state_lstm_concat')  # internal state of the cell (before output gate)
		self.hidden_state_lstm_concat = tf.placeholder(dtype=tf.float32, shape=[None, self.rnn_state_size_lstm_concat],
		                                               name='hidden_state_lstm_concat')  # output of the cell (after output gate)
		self.init_state_tuple_lstm_concat = tf.contrib.rnn.LSTMStateTuple(self.cell_state_lstm_concat,
		                                                                  self.hidden_state_lstm_concat)

		inputs_series = tf.unstack(self.input_state_placeholder, axis=1)
		outputs_series = tf.unstack(self.output_placeholder, axis=1)

		# Print network info
		print("Input list length: {}".format(len(inputs_series)))
		print("Output list length: {}".format(len(outputs_series)))
		print("Single input shape: {}".format(inputs_series[0].get_shape()))
		print("Single output shape: {}".format(outputs_series[0].get_shape()))

		# Set up model with variables
		with tf.variable_scope('model') as scope:

			# CNN weights for static occupancy grid processing
			train_grid_encoder_conv = False
			train_grid_encoder_fc = False
			use_summary_convnet = False

			# Convolutional layer 1
			self.conv1_kernel_size = 5
			self.conv1_number_filters = 64
			self.conv1_stride_length = 2
			self.conv1_weights = self.get_weight_variable(name="conv1_weights",
			                                              shape=[self.conv1_kernel_size, self.conv1_kernel_size, 1,
			                                                     self.conv1_number_filters],
			                                              trainable=train_grid_encoder_conv, summary=use_summary_convnet)
			self.conv1_biases = self.get_bias_variable(name="conv1_biases", shape=[self.conv1_number_filters],
			                                           trainable=train_grid_encoder_conv, summary=use_summary_convnet)

			# Convolutional layer 2
			self.conv2_kernel_size = 3
			self.conv2_number_filters = 32
			self.conv2_stride_length = 2
			self.conv2_weights = self.get_weight_variable(name="conv2_weights",
			                                              shape=[self.conv2_kernel_size, self.conv2_kernel_size,
			                                                     self.conv1_number_filters, self.conv2_number_filters],
			                                              trainable=train_grid_encoder_conv, summary=use_summary_convnet)
			self.conv2_biases = self.get_bias_variable(name="conv2_biases", shape=[self.conv2_number_filters],
			                                           trainable=train_grid_encoder_conv, summary=use_summary_convnet)

			# Convolutional layer 3
			self.conv3_kernel_size = 3
			self.conv3_number_filters = 8
			self.conv3_stride_length = 2
			self.conv3_weights = self.get_weight_variable(name="conv3_weights",
			                                              shape=[self.conv3_kernel_size, self.conv3_kernel_size,
			                                                     self.conv2_number_filters, self.conv3_number_filters],
			                                              trainable=train_grid_encoder_conv, summary=use_summary_convnet)
			self.conv3_biases = self.get_bias_variable(name="conv3_biases", shape=[self.conv3_number_filters],
			                                           trainable=train_grid_encoder_conv, summary=use_summary_convnet)

			fc_grid_hidden_dim = 64
			self.fc_grid_weights = self.get_weight_variable(shape=[512, fc_grid_hidden_dim], name="fc_grid_weights",
			                                                trainable=train_grid_encoder_fc)
			self.fc_grid_biases = self.get_bias_variable(shape=[fc_grid_hidden_dim], name="fc_grid_biases",
			                                             trainable=train_grid_encoder_fc)
			with tf.variable_scope('encoder_model') as scope:
				# Pedestrian grid embedding layer
				ped_grid_out_dim = 128
				self.W_pedestrian_grid = self.get_weight_variable(name='w_ped_grid',
				                                                  shape=[self.pedestrian_vector_dim, ped_grid_out_dim],
				                                                  trainable=self.is_training)
				self.b_pedestrian_grid = self.get_bias_variable(name='b_ped_grid', shape=[1, ped_grid_out_dim],
				                                                trainable=self.is_training)

				###################### Assemble Neural Net #####################
				# Process occupancy grid, apply convolutional filters
				occ_grid_series = tf.unstack(self.input_grid_placeholder, axis=1)
				conv_grid_feature_series = self.process_grid(occ_grid_series, trainable=self.is_training)

				# Process pedestrian grid
				ped_grid_series = tf.unstack(self.input_ped_grid_placeholder, axis=1)
				# Embedding layer of pedestrian grid
				ped_grid_feature_series = self.process_pedestrian_grid(ped_grid_series)

				# LSTM cells to process static occupancy grid
				with tf.variable_scope('lstm_grid') as scope:
					cell_grid = tf.keras.layers.LSTMCell(self.rnn_state_size_lstm_grid, name='basic_lstm_cell')
					self.cell_outputs_series_lstm_grid, self.current_state_lstm_grid = tf.contrib.rnn.static_rnn(cell_grid,
					                                                                                             conv_grid_feature_series,
					                                                                                             dtype=tf.float32,
					                                                                                             initial_state=self.init_state_tuple_lstm_grid)

				# LSTM cells to process pedestrian grid
				with tf.variable_scope('lstm_ped') as scope:
					cell_ped = tf.keras.layers.LSTMCell(self.rnn_state_size_lstm_ped, name='basic_lstm_cell')
					self.cell_outputs_series_lstm_ped, self.current_state_lstm_ped = tf.contrib.rnn.static_rnn(cell_ped,
					                                                                                           ped_grid_feature_series,
					                                                                                           dtype=tf.float32,
					                                                                                           initial_state=self.init_state_tuple_lstm_ped)

				# LSTM cells to process pedestrian state
				with tf.variable_scope('lstm_state') as scope:
					cell = tf.keras.layers.LSTMCell(self.rnn_state_size, name='basic_lstm_cell')
					self.cell_outputs_series_state, self.current_state = tf.contrib.rnn.static_rnn(cell,
					                                                                               inputs_series,
					                                                                               dtype=tf.float32,
					                                                                               initial_state=self.init_state_tuple)

				# Concatenate the outputs of the three previous LSTMs and feed in first concatenated LSTM cell
				concat_lstm = []
				for lstm_out, grid_feature, ped_feature in zip(self.cell_outputs_series_state, self.cell_outputs_series_lstm_grid,
				                                               self.cell_outputs_series_lstm_ped):
					concat_lstm.append(
						tf.concat([lstm_out, grid_feature, ped_feature], axis=1, name="concatenate_lstm_out_with_grid_and_ped"))

			with tf.variable_scope('decoder_model') as scope:
				# Hidden layer (fully connected) after concatenation LSTM
				W_fc = self.get_weight_variable(name='w_fc',
				                                shape=[self.rnn_state_size_lstm_concat, self.fc_hidden_unit_size],
				                                trainable=self.is_training)
				b_fc = self.get_bias_variable(name='b_fc', shape=[1, self.fc_hidden_unit_size],
				                              trainable=self.is_training)

				# Output layer (fully connected)
				W_out = self.get_weight_variable(name='w_out', shape=[self.fc_hidden_unit_size, self.output_dim],
				                                 trainable=self.is_training)
				b_out = self.get_bias_variable(name='b_out', shape=[1, self.output_dim], trainable=self.is_training)

				# Second concatenated LSTM layer
				with tf.variable_scope('lstm_concat') as scope:
					cell_lstm_concat = tf.keras.layers.LSTMCell(self.rnn_state_size_lstm_concat, name='basic_lstm_cell')
					self.cell_outputs_series_lstm_concat, self.current_state_lstm_concat = tf.contrib.rnn.static_rnn(
						cell_lstm_concat, concat_lstm, dtype=tf.float32, initial_state=self.init_state_tuple_lstm_concat)

				# FC layers
				fc_fuse = [tf.nn.relu(tf.add(tf.matmul(lstm_out, W_fc), b_fc)) for lstm_out in
				           self.cell_outputs_series_lstm_concat]
				fc_drop = [tf.layers.dropout(inputs=dense, rate=1-self.keep_prob) for dense in
				           fc_fuse]
				# fc_fuse2 = [tf.nn.elu(tf.add(tf.matmul(fc_out, W_fc2), b_fc2)) for fc_out in fc_fuse]
				# Model prediction
				self.prediction = [tf.add(tf.matmul(fc_out, W_out), b_out) for fc_out in fc_fuse]

			# Loss function
			loss_list = []
			lossfunc = []
			# Iterate through tbp steps
			for pred, out in zip(self.prediction,
			                     outputs_series):  # prediction dimension: output_sequence_length * output_states (e.g. [x_1, y_1, x_2, y_2, x_3, y_3])
				prediction_loss_list = []
				val_prediction_loss_list = []
				for prediction_step in range(self.output_sequence_length):
					# Compute Euclidean error for each prediction step within each tbp step
					idx = prediction_step * self.output_pred_state_dim * self.n_mixtures
					idx_x = prediction_step * self.output_state_dim
					idx_y = idx_x + 1

					# Calculate loss for the current ped
					lossfunc= tf.sqrt(tf.add(tf.square(tf.subtract(pred[:, idx_x], out[:, idx_x])),  # error in x
						               tf.square(tf.subtract(pred[:, idx_y], out[:, idx_y]))))  # error in y

					prediction_loss_list.append(lossfunc)  # sum over number of batches

				loss_list.append(tf.reduce_mean(prediction_loss_list)) # mean over prediction horizon

			# Get all trainable variables
			tvars = tf.trainable_variables()

			# L2 loss
			l2 = self.lambda_ * sum(tf.nn.l2_loss(tvar) for tvar in tvars)

			# Reduce mean in all dimensions
			self.total_loss = tf.reduce_mean(loss_list, axis=0) #+ l2 # mean over tbp steps

			# Parameter optimization

			# Exponential / stepwise decrease of learning rate
			global_step = tf.Variable(0, trainable=False)
			self.learning_rate = tf.train.exponential_decay(args.learning_rate_init, global_step, decay_steps=5000,
			                                                decay_rate=0.975, staircase=True)
			self.beta = (tf.tanh((tf.to_float(self.step) - 3500) / 1000) + 1) / 2
			# Optimizer specification
			#self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
			self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
			# Compute gradients
			train_vars = tf.trainable_variables()
			self.gradients = tf.gradients(self.total_loss, train_vars)

			# Clip gradients
			self.grads, _ = tf.clip_by_global_norm(self.gradients, self.grads_clip)

			# Apply gradients to update model
			self.update = self.optimizer.apply_gradients(zip(self.grads, train_vars), global_step=global_step)
			# self.update = self.optimizer.minimize(self.total_loss, global_step=global_step)
			# Tensorboard summary

			## Loss
			tf.summary.scalar('loss', self.total_loss)
			tf.summary.scalar('learning_rate', self.learning_rate)
			self.summary = tf.summary.merge_all()
			self.summary_writer = tf.summary.FileWriter(logdir=os.path.join(args.log_dir, 'train'), graph=tf.Session().graph)

			# Saver

			# Save / restore all variables
			self.full_saver = tf.train.Saver()

			# Only save / restore the parameters responsible for occupancy grid processing
			self.convnet_saver = tf.train.Saver(var_list={'conv1_weights': self.conv1_weights,
			                                              'conv1_biases': self.conv1_biases,
			                                              'conv2_weights': self.conv2_weights,
			                                              'conv2_biases': self.conv2_biases,
			                                              'conv3_weights:': self.conv3_weights,
			                                              'conv3_biases': self.conv3_biases,
			                                              'fc_grid_weights': self.fc_grid_weights,
			                                              'fc_grid_biases': self.fc_grid_biases,
			                                              })
			# Retrieve just the LSTM variables.
			lstm_variables = [v for v in tf.all_variables()
			                  if "encoder" in v.name]
			self.encoder_saver = tf.train.Saver(var_list=lstm_variables)

	def feed_dic(self, batch_x, batch_grid, batch_ped_grid, step, batch_y):
		return {self.input_state_placeholder: batch_x[:, :, 2:],
		        self.input_grid_placeholder: batch_grid,
		        self.input_ped_grid_placeholder: batch_ped_grid,
		        self.step: step,
		        self.output_placeholder: batch_y,
		        self.cell_state: self.cell_state_current,
		        self.hidden_state: self.hidden_state_current,
		        self.cell_state_lstm_grid: self.cell_state_current_lstm_grid,
		        self.hidden_state_lstm_grid: self.hidden_state_current_lstm_grid,
		        self.cell_state_lstm_ped: self.cell_state_current_lstm_ped,
		        self.hidden_state_lstm_ped: self.hidden_state_current_lstm_ped,
		        self.cell_state_lstm_concat: self.cell_state_current_lstm_concat,
						self.hidden_state_lstm_concat: self.hidden_state_current_lstm_concat
		        }

	def feed_test_dic(self, batch_x, batch_grid, batch_ped_grid,batch_target):
		return {self.input_state_placeholder: batch_x,
		        self.input_grid_placeholder: batch_grid,
		        self.input_ped_grid_placeholder: batch_ped_grid,
		        self.output_placeholder: batch_target,
		        self.cell_state: self.cell_state_current,
		        self.hidden_state: self.hidden_state_current,
		        self.cell_state_lstm_grid: self.cell_state_current_lstm_grid,
		        self.hidden_state_lstm_grid: self.hidden_state_current_lstm_grid,
		        self.cell_state_lstm_ped: self.cell_state_current_lstm_ped,
		        self.hidden_state_lstm_ped: self.hidden_state_current_lstm_ped,
		        self.cell_state_lstm_concat: self.cell_state_current_lstm_concat,
		        self.hidden_state_lstm_concat: self.hidden_state_current_lstm_concat,
		        }

	def run(self, sess, feed_dict_train):
		_, batch_loss, _current_state, _current_state_lstm_grid, _current_state_lstm_ped, \
		_current_state_lstm_concat, \
		_model_prediction, _summary_str, lr, beta = sess.run([self.update,
		                                                      self.total_loss,
		                                                      self.current_state,
		                                                      self.current_state_lstm_grid,
		                                                      self.current_state_lstm_ped,
		                                                      self.current_state_lstm_concat,
		                                                      self.prediction,
		                                                      self.summary,
		                                                      self.learning_rate,
		                                                      self.beta],
		                                                     feed_dict=feed_dict_train)
		self.cell_state_current, self.hidden_state_current = _current_state
		self.cell_state_current_lstm_grid, self.hidden_state_current_lstm_grid = _current_state_lstm_grid
		self.cell_state_current_lstm_ped, self.hidden_state_current_lstm_ped = _current_state_lstm_ped
		self.cell_state_current_lstm_concat, self.hidden_state_current_lstm_concat = _current_state_lstm_concat

		return batch_loss, _model_prediction, _summary_str, lr, beta

	def predict(self, sess, feed_dict_train,update_state=True):
		_current_state, _current_state_lstm_grid, _current_state_lstm_ped, \
		_current_state_lstm_concat, \
		_model_prediction, loss = sess.run([self.current_state,
		                              self.current_state_lstm_grid,
		                              self.current_state_lstm_ped,
		                              self.current_state_lstm_concat,
		                              self.prediction,
		                              self.total_loss],
		                             feed_dict=feed_dict_train)
		if update_state:
			self.cell_state_current, self.hidden_state_current = _current_state
			self.cell_state_current_lstm_grid, self.hidden_state_current_lstm_grid = _current_state_lstm_grid
			self.cell_state_current_lstm_ped, self.hidden_state_current_lstm_ped = _current_state_lstm_ped
			self.cell_state_current_lstm_concat, self.hidden_state_current_lstm_concat = _current_state_lstm_concat

		return _model_prediction, loss

	def reset_cells(self,sequence_reset):
		# Initialize the new sequences with a hidden state of zeros, the continuing sequences get assigned the previous hidden state
		if np.any(sequence_reset):
			for sequence_idx in range(sequence_reset.shape[0]):
				if sequence_reset[sequence_idx] == 1:
					# Hidden state of specific batch entry will be initialized with zeros if there was a jump in sequence
					self.cell_state_current[sequence_idx, :] = np.zeros([1, self.rnn_state_size])
					self.hidden_state_current[sequence_idx, :] = np.zeros([1, self.rnn_state_size])
					self.cell_state_current_lstm_grid[sequence_idx, :] = np.zeros([1, self.rnn_state_size_lstm_grid])
					self.hidden_state_current_lstm_grid[sequence_idx, :] = np.zeros([1, self.rnn_state_size_lstm_grid])
					self.cell_state_current_lstm_ped[sequence_idx, :] = np.zeros([1, self.rnn_state_size_lstm_ped])
					self.hidden_state_current_lstm_ped[sequence_idx, :] = np.zeros([1, self.rnn_state_size_lstm_ped])
					self.cell_state_current_lstm_concat[sequence_idx, :] = np.zeros([1, self.rnn_state_size_lstm_concat])
					self.hidden_state_current_lstm_concat[sequence_idx, :] = np.zeros([1, self.rnn_state_size_lstm_concat])

	def update_states(self, current_state, current_state_lstm_grid, current_state_lstm_ped, current_state_lstm_concat):
		# Store the hidden states to re-initialize the in the next training iteration if necessary
		self.cell_state_current, self.hidden_state_current = current_state
		self.cell_state_current_lstm_grid, self.hidden_state_current_lstm_grid = current_state_lstm_grid
		self.cell_state_current_lstm_ped, self.hidden_state_current_lstm_ped = current_state_lstm_ped
		self.cell_state_current_lstm_concat, self.hidden_state_current_lstm_concat = current_state_lstm_concat

	def warmstart_model(self, args, sess):
		# Restore whole model
		print('Loading session from "{}"'.format(args.model_path))
		ckpt = tf.train.get_checkpoint_state(args.model_path)
		print('Restoring model {}'.format(ckpt.model_checkpoint_path))
		self.full_saver.restore(sess, ckpt.model_checkpoint_path)

	def warmstart_convnet(self, args, sess):
		# Restore convnet parameters from pretrained ones (autoencoder encoding part)
		print('Loading convnet parameters from "{}"'.format(args.pretrained_convnet_path))
		ckpt_conv = tf.train.get_checkpoint_state(args.pretrained_convnet_path)
		if ckpt_conv == None:
			print("Error. Convnet not finded...")
			exit()
		else:
			print('Restoring convnet {}'.format(ckpt_conv.model_checkpoint_path))
			self.convnet_saver.restore(sess, ckpt_conv.model_checkpoint_path)

	def process_grid(self, occ_grid_series, trainable):
		"""
		Process occupancy grid with series of convolutional and fc layers.
		input: occupancy grid series (shape: tbpl * [batch_size, grid_dim_x, grid_dim_y])
		output: convolutional feature vector (shape: list of tbpl elements with [batch_size, feature_vector_size] each)
		"""

		conv_feature_vectors = []
		print("Length occ grid series: {}".format(len(occ_grid_series)))
		for idx, grid_batch in enumerate(occ_grid_series):
			grid_batch = tf.expand_dims(input=grid_batch, axis=3)
			conv1 = self.conv_layer(input=grid_batch, weights=self.conv1_weights, biases=self.conv1_biases,
			                        conv_stride_length=self.conv1_stride_length, name="conv1_grid")
			conv2 = self.conv_layer(input=conv1, weights=self.conv2_weights, biases=self.conv2_biases,
			                        conv_stride_length=self.conv2_stride_length, name="conv2_grid")
			conv3 = self.conv_layer(input=conv2, weights=self.conv3_weights, biases=self.conv3_biases,
			                        conv_stride_length=self.conv3_stride_length, name="conv3_grid")
			conv_grid_size = 8
			conv5_flat = tf.reshape(conv3, [-1, conv_grid_size * conv_grid_size * self.conv3_number_filters])
			fc_final = self.fc_layer(input=conv5_flat, weights=self.fc_grid_weights, biases=self.fc_grid_biases,
			                         use_activation=True, name="fc_grid")
			if idx == 0:
				tf.summary.histogram("fc_activations", fc_final)

			# Flatten to obtain feature vector
			conv_features = tf.contrib.layers.flatten(fc_final)
			conv_feature_vectors.append(conv_features)

		return conv_feature_vectors

	def process_pedestrian_grid(self, ped_grid_series):
		"""
		Process pedestrian grid with a FC layer
		"""
		fc_feature_vector = []
		for idx, grid_batch in enumerate(ped_grid_series):
			grid_batch = tf.contrib.layers.flatten(grid_batch)
			fc_feature_vector.append(
				self.fc_layer(grid_batch, weights=self.W_pedestrian_grid, biases=self.b_pedestrian_grid, use_activation=False,
				              name="fc_ped"))
			if idx == 0:
				tf.summary.histogram("fc_ped_activations", fc_feature_vector)
		#       fc_feature_vector.append(grid_batch)
		return fc_feature_vector

	def get_weight_variable(self, shape, name, trainable=True, regularizer=None, summary=True):
		"""
		Get weight variable with specific initializer.
		"""
		var = tf.get_variable(name=name, shape=shape, initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1),
		                      regularizer=tf.contrib.layers.l2_regularizer(0.01), trainable=trainable)
		if summary:
			tf.summary.histogram(name, var)
		return var

	def get_bias_variable(self, shape, name, trainable=True, regularizer=None, summary=True):
		"""
		Get bias variable with specific initializer.
		"""
		var = tf.get_variable(name=name, shape=shape, initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1),
		                      trainable=trainable)
		if summary:
			tf.summary.histogram(name, var)
		return var

	def get_bayesian_weight_variable(self, shape, name, trainable=True, regularizer=None, summary=True):
		"""
		Get weight variable with specific initializer.
		"""
		var = tf.Variable(tf.truncated_normal(shape, stddev=0.1), dtype=tf.float32, name=name)
		if summary:
			tf.summary.histogram(name, var)
		return var

	def get_bayesian_bias_variable(self, shape, name, trainable=True, regularizer=None, summary=True):
		"""
		Get bias variable with specific initializer.
		"""
		var = tf.Variable(tf.truncated_normal(shape, stddev=0.1), dtype=tf.float32, name=name)
		if summary:
			tf.summary.histogram(name, var)
		return var

	def conv_layer(self, input, weights, biases, conv_stride_length=1, padding="SAME", name="conv", summary=False):
		"""
		Convolutional layer including a ReLU activation but excluding pooling.
		"""
		conv = tf.nn.conv2d(input, filter=weights, strides=[1, conv_stride_length, conv_stride_length, 1], padding=padding,
		                    name=name)
		activations = tf.nn.relu(conv + biases)
		if summary:
			tf.summary.histogram(name, activations)
		return activations

	def fc_layer(self, input, weights, biases, use_activation=False, name="fc", summary=False):
		"""
		Fully connected layer with given weights and biases.
		Activation and summary can be activated with the arguments.
		"""
		affine_result = tf.matmul(input, weights) + biases
		if use_activation:
			activations = tf.nn.sigmoid(affine_result)
		else:
			activations = affine_result
		if summary:
			tf.summary.histogram(name + "_activations", activations)
		return activations




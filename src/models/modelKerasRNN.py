import tensorflow as tf
import numpy as np
import os
import sys
sys.path.append('./autoencoder_grid')
tfk = tf.keras
tfkl = tf.keras.layers

class EncoderState(tfk.Model):
  def __init__(self, rnn_state_size,truncated_backprop_length, input_state_dim,batch_size):
    super(EncoderState, self).__init__()

    self.rnn_state_size = rnn_state_size
    self.truncated_backprop_length = truncated_backprop_length
    self.input_state_dim = input_state_dim
    self.batch_size = batch_size
    self.lambda_ = 0.01

    self.agent_state_lstm = tfkl.GRU(self.rnn_state_size, input_shape=[self.truncated_backprop_length, self.input_state_dim],
                                 name='ped_lstm',return_sequences=True,
                                 return_state=True,
                                 kernel_regularizer=tfk.regularizers.l2(l=self.lambda_))

  def call(self, x, hidden):
    output, state = self.agent_state_lstm(x, initial_state = hidden)
    return output, state

  def initialize_hidden_state(self):
    return tf.zeros((self.batch_size, self.rnn_state_size))

class EncoderPeds(tfk.Model):
  def __init__(self, rnn_state_size_lstm_ped, truncated_backprop_length, input_state_dim, batch_size):
    super(EncoderPeds, self).__init__()

    self.rnn_state_size_lstm_ped = rnn_state_size_lstm_ped
    self.truncated_backprop_length = truncated_backprop_length
    self.input_state_dim = input_state_dim
    self.lambda_ = 0.01
    self.batch_size = batch_size
    self.ped_embed = tfkl.Dense(self.rnn_state_size_lstm_ped, activation='relu')

    self.agent_other_peds_lstm = tfkl.GRU(self.rnn_state_size_lstm_ped,
                                     input_shape=[self.truncated_backprop_length, self.input_state_dim],
                                     name='other_peds_lstm', return_sequences=True,
                                     return_state=True,
                                     kernel_regularizer=tfk.regularizers.l2(l=self.lambda_))

  def call(self, x, hidden):
    x = self.ped_embed(x)
    output, state = self.agent_other_peds_lstm(x, initial_state=hidden)
    return output, state

  def initialize_hidden_state(self):
    return tf.zeros((self.batch_size, self.rnn_state_size_lstm_ped))

class BahdanauAttention(tfkl.Layer):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, query, values):
    # hidden shape == (batch_size, hidden size)
    # hidden_with_time_axis shape == (batch_size, 1, hidden size)
    # we are doing this to perform addition to calculate the score
    hidden_with_time_axis = tf.expand_dims(query, 1)

    # score shape == (batch_size, max_length, 1)
    # we get 1 at the last axis because we are applying score to self.V
    # the shape of the tensor before applying self.V is (batch_size, max_length, units)
    score = self.V(tf.nn.tanh(
        self.W1(values) + self.W2(hidden_with_time_axis)))

    # attention_weights shape == (batch_size, max_length, 1)
    attention_weights = tf.nn.softmax(score, axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights

def scaled_dot_product_attention(q, k, v, mask):
  """Calculate the attention weights.
  q, k, v must have matching leading dimensions.
  k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
  The mask has different shapes depending on its type(padding or look ahead)
  but it must be broadcastable for addition.

  Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable
          to (..., seq_len_q, seq_len_k). Defaults to None.

  Returns:
    output, attention_weights
  """

  matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

  # scale matmul_qk
  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

  # add the mask to the scaled tensor.
  if mask is not None:
    scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
  # add up to 1.
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

  output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

  return output, attention_weights

class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model

    assert d_model % self.num_heads == 0

    self.depth = d_model // self.num_heads

    self.wq = tf.keras.layers.Dense(d_model)
    self.wk = tf.keras.layers.Dense(d_model)
    self.wv = tf.keras.layers.Dense(d_model)

    self.dense = tf.keras.layers.Dense(d_model)

  def split_heads(self, x, batch_size):
    """Split the last dimension into (num_heads, depth).
    Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
    """
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])

  def call(self, v, k, q, mask):
    batch_size = tf.shape(q)[0]

    q = self.wq(q)  # (batch_size, seq_len, d_model)
    k = self.wk(k)  # (batch_size, seq_len, d_model)
    v = self.wv(v)  # (batch_size, seq_len, d_model)

    q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
    k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
    v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

    # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
    # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
    scaled_attention, attention_weights = scaled_dot_product_attention(
      q, k, v, mask)

    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

    concat_attention = tf.reshape(scaled_attention,
                                  (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

    output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

    return output, attention_weights

class Decoder(tfk.Model):
  def __init__(self, args):
    super(Decoder, self).__init__()
    self.batch_size = args.batch_size
    self.truncated_backprop_length = args.truncated_backprop_length
    self.prediction_horizon = args.prediction_horizon
    self.rnn_state_size_lstm_concat = args.rnn_state_size_lstm_concat
    self.fc_hidden_unit_size = args.fc_hidden_unit_size
    self.output_dim = args.output_dim
    self.lambda_ = 0.01

    self.gru = tf.keras.layers.GRU(self.rnn_state_size_lstm_concat,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.fc1 = tfkl.Dense(self.fc_hidden_unit_size, activation='relu', name='concat_fc',
                                              kernel_regularizer=tfk.regularizers.l2(l=self.lambda_))
    self.fco = tfkl.Dense(self.output_dim*self.prediction_horizon, activation='linear', name='out_fc',
                                              kernel_regularizer=tfk.regularizers.l2(l=self.lambda_))

    # used for attention
    self.attention = BahdanauAttention(self.rnn_state_size_lstm_concat)

  def call(self, x, hidden,ped_info):

    # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
    concat_x = tf.concat([ped_info, x], axis=-1)

    context_vector, attention_weights = self.attention(hidden, concat_x)

    context_vector = tf.expand_dims(context_vector, 1)

    # passing the concatenated vector to the GRU
    output, state = self.gru(concat_x,initial_state=hidden)

    # two FCL
    output = self.fc1(output)
    output = self.fco(output)

    return output, state, attention_weights

  def initialize_hidden_state(self):
    return tf.zeros((self.batch_size, self.rnn_state_size_lstm_concat))

class RNN(tf.keras.Model):

  def __init__(self,args):
    super(RNN, self).__init__()
    self.args = args

    # Network Elements

    self.agent_state_encoder = EncoderState(self.args.rnn_state_size, self.args.truncated_backprop_length, self.args.input_state_dim, self.args.batch_size)
    self.ped_info_encoder = EncoderPeds(self.args.rnn_state_size_lstm_ped, self.args.truncated_backprop_length, self.args.input_state_dim, self.args.batch_size)
    self.decoder = Decoder(self.args)
    # Initialize States
    self.state_hidden = self.agent_state_encoder.initialize_hidden_state()
    self.ped_info_hidden = self.ped_info_encoder.initialize_hidden_state()
    self.dec_hidden = self.decoder.initialize_hidden_state()

    # Optimizer and Loss
    self.optimizer = tf.keras.optimizers.Adam()
    self.loss_object = tf.keras.losses.Huber()

  def call(self, x, ped_info):
    enc_state_output, self.state_hidden = self.agent_state_encoder(x, self.state_hidden)

    enc_ped_output, self.ped_info_hidden = self.ped_info_encoder(ped_info, self.ped_info_hidden)

    return self.decoder(enc_state_output, self.dec_hidden,enc_ped_output)

  def loss_function(self,real, pred):
    loss_ = self.loss_object(real, pred)

    return tf.reduce_mean(loss_)

  @tf.function
  def train_step(self,x, ped, targ):
    loss = 0

    with tf.GradientTape() as tape:
      enc_state_output, enc_state_hidden = self.agent_state_encoder(x, self.state_hidden)

      enc_ped_output, enc_ped_info_hidden = self.ped_info_encoder(ped, self.ped_info_hidden)

      predictions, dec_hidden, attention_weights = self.decoder(enc_state_output, self.dec_hidden,
                                                           enc_ped_output)

      # Teacher forcing - feeding the target as the next input
      for t in range(1, targ.shape[1]):
        loss += self.loss_function(targ[:, t], predictions[:, t])

    batch_loss = (loss / int(targ.shape[1]))

    gradients = tape.gradient(loss, self.trainable_variables)

    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    return batch_loss, enc_state_hidden, enc_ped_info_hidden, dec_hidden, predictions

  @tf.function
  def predict(self,batch_x, other_agents_info):
    enc_state_output, enc_state_hidden = self.agent_state_encoder(batch_x, self.state_hidden)

    enc_ped_output, enc_ped_info_hidden = self.ped_info_encoder(other_agents_info, self.ped_info_hidden)

    predictions, dec_hidden, attention_weights = self.decoder(enc_state_output, self.dec_hidden,
                                                              enc_ped_output)

    return predictions
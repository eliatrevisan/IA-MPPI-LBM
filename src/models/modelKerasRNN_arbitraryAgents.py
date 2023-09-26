import tensorflow as tf
tfk = tf.keras
tfkm = tf.keras.models
tfkl = tf.keras.layers

class StateEncoder(tfkl.Layer):
    def __init__(self, args):
        super().__init__()
        
        self.lambda_ = 0.01
        self.rnn_state_size = args.rnn_state_size
        
        self.lstm_ped = tfkl.LSTM(self.rnn_state_size, name = 'lstm_state', return_sequences = False, kernel_regularizer=tfk.regularizers.l2(l=self.lambda_))
        
    def call(self, x):
        x = self.lstm_ped(x)
        return x

class OtherPedsEncoder(tfkl.Layer):
    def __init__(self, args):
        super().__init__()

        self.lambda_ = 0.01
        self.rnn_state_size_lstm_ped = args.rnn_state_size_lstm_ped
        self.rnn_state_size_bilstm = args.rnn_state_size_lstm_ped # TODO: Possibly add another argument to modify this state size

        self.lstm_other_peds = tfkl.LSTM(self.rnn_state_size_lstm_ped, name = 'lstm_other_peds', return_sequences = False, kernel_regularizer=tfk.regularizers.l2(l=self.lambda_))
        self.bilstm = tfkl.Bidirectional(tfkl.LSTM(self.rnn_state_size_bilstm,  name = 'bilstm_other_peds', return_sequences = False, return_state = False, kernel_regularizer=tfk.regularizers.l2(l=self.lambda_)), merge_mode = 'ave') # Average of forward and backward LSTMs
        
    def call(self, x):
        other_peds = []
        for x_ped in x:
            other_peds.append(self.lstm_other_peds(x_ped))
        stacked_other_peds = tf.stack(other_peds, axis = 1)
        out = self.bilstm(stacked_other_peds)
        return out

class Decoder(tfkl.Layer):
    def __init__(self, args):
        super().__init__()
        
        self.output_dim = args.output_dim
    
        self.rnn_state_size_lstm_concat = args.rnn_state_size_lstm_concat
        self.fc_hidden_unit_size = args.fc_hidden_unit_size

        self.lstm_concat = tfkl.LSTM(self.rnn_state_size_lstm_concat, name = 'lstm_decoder', return_sequences = True, return_state = False)
        self.fc1 = tfkl.TimeDistributed(tfkl.Dense(self.fc_hidden_unit_size, name = 'fc_decoder', activation = 'relu'))
        self.fco = tfkl.TimeDistributed(tfkl.Dense(self.output_dim, name = 'fc_out'))
                
    def call(self, x):        
        x = self.lstm_concat(x)
        x = self.fc1(x)
        x = self.fco(x)
        return x

class NetworkModel(tfk.Model):
    def __init__(self, args):
        super().__init__()
        
        future_steps = args.prediction_horizon
        
        # Define optimizer and loss
        self.loss_object = tfk.losses.MeanSquaredError()
        self.optimizer = tfk.optimizers.Adam()
        self.train_loss = tfk.metrics.MeanSquaredError(name='train_loss')
        self.val_loss = tfk.metrics.MeanSquaredError(name='val_loss')
        
        # Define architecture
        self.state_encoder = StateEncoder(args)
        self.other_peds_encoder = OtherPedsEncoder(args)
        self.concat = tfkl.Concatenate()
        self.repeat = tfkl.RepeatVector(future_steps)
        self.decoder = Decoder(args)
                
    def call(self, x):
        x1 = self.state_encoder(x[0])
        
        x2 = self.other_peds_encoder(x[1:])
        concat = self.concat([x1, x2])
        repeated = self.repeat(concat)
        out = self.decoder(repeated)
        return out
    
    @tf.function
    def train_step(self, X, Y):
        with tf.GradientTape() as tape:
            predictions = self(X)
            loss = self.loss_object(Y, predictions)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.train_loss(Y, predictions)

    @tf.function
    def val_step(self, X, Y):
        predictions = self(X)
        v_loss = self.loss_object(Y, predictions)

        self.val_loss(Y, predictions)



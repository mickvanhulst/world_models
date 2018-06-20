import sys
import math
import numpy as np

from keras.layers import Input, LSTM, Dense
from keras.models import Model
from keras import backend as K
from keras.callbacks import EarlyStopping
import config
import tensorflow as tf

HIDDEN_UNITS = 256

def get_mixture_coef(y_pred):
    d = config.GAUSSIAN_MIXTURES * config.Z_DIM
    rollout_length = K.shape(y_pred)[1]

    pi = y_pred[:, :, :d]
    mu = y_pred[:, :, d:(2 * d)]
    log_sigma = y_pred[:, :, (2 * d):(3 * d)]

    pi = K.reshape(pi, [-1, rollout_length, config.GAUSSIAN_MIXTURES, config.Z_DIM])
    mu = K.reshape(mu, [-1, rollout_length, config.GAUSSIAN_MIXTURES, config.Z_DIM])
    log_sigma = K.reshape(log_sigma, [-1, rollout_length, config.GAUSSIAN_MIXTURES, config.Z_DIM])

    pi = K.exp(pi) / K.sum(K.exp(pi), axis=2, keepdims=True)
    sigma = K.exp(log_sigma)

    return pi, mu, sigma  # , discrete


def tf_normal(y_true, mu, sigma, pi):
    rollout_length = K.shape(y_true)[1]
    y_true = K.tile(y_true, (1, 1, config.GAUSSIAN_MIXTURES))
    y_true = K.reshape(y_true, [-1, rollout_length, config.GAUSSIAN_MIXTURES, config.Z_DIM])

    oneDivSqrtTwoPI = 1 / math.sqrt(2 * math.pi)
    result = y_true - mu

    result = result * (1 / (sigma + 1e-8))
    result = -K.square(result) / 2
    result = K.exp(result) * (1 / (sigma + 1e-8)) * oneDivSqrtTwoPI
    result = result * pi
    result = K.sum(result, axis=2)

    return result

class RNN():
    def __init__(self):
        self.models = self._build()
        self.model = self.models[0]
        self.forward = self.models[1]
        self.hidden_units = HIDDEN_UNITS

    def _build(self):
        # Training model
        input_x = Input(shape=(None, config.Z_DIM + config.ACTION_DIM))
        lstm = LSTM(HIDDEN_UNITS, return_sequences=True, return_state=True)

        lstm_output, _, _ = lstm(input_x)
        mdn = Dense(config.GAUSSIAN_MIXTURES * (3 * config.Z_DIM))(lstm_output)  # + discrete_dim

        rnn = Model(input_x, mdn)

        # Predictive model
        input_h = Input(shape=(HIDDEN_UNITS,))
        input_c = Input(shape=(HIDDEN_UNITS,))
        inputs = [input_h, input_c]
        _, state_h, state_c = lstm(input_x, initial_state=[input_h, input_c])

        forward = Model([input_x] + inputs, [mdn, state_h, state_c])

        def r_loss(y_true, y_pred):
            pi, mu, sigma = get_mixture_coef(y_pred)

            res = tf_normal(y_true, mu, sigma, pi)

            res = -K.log(res + 1e-8)
            res = K.mean(res, axis=(1, 2))
            return res

        def kl_loss(y_true, y_pred):
            pi, mu, sigma = get_mixture_coef(y_pred)
            kl_loss = - 0.5 * K.mean(1 + K.log(K.square(sigma)) - K.square(mu) - K.square(sigma), axis=[1, 2, 3])
            return kl_loss

        def loss_func(y_true, y_pred):
            return r_loss(y_true, y_pred)

        rnn.compile(loss=loss_func, optimizer='rmsprop', metrics=[r_loss, kl_loss])

        return (rnn, forward)

    def set_weights(self, filepath):
        self.model.load_weights(filepath)

    def train(self, rnn_input, rnn_output, validation_split=0.2):
        earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5, verbose=2, mode='auto')
        callbacks_list = [earlystop]
        print('--------------')
        self.model.fit(rnn_input, rnn_output,
                       shuffle=True,
                       epochs=config.RNN_EPOCHS,
                       batch_size=config.RNN_BATCH_SIZE,
                       validation_split=validation_split,
                       callbacks=callbacks_list)

        self.model.save_weights('./weights/rnn/weights_' + sys.argv[1] + '.h5')

    def save_weights(self, filepath):
        self.model.save_weights(filepath)

def main():
    rnn = RNN()

    if not config.NEW_MODEL:
        rnn.set_weights('./weights/rnn/weights_' + sys.argv[1] + '.h5')

    for batch_num in range(config.START_BATCH, config.MAX_BATCH + 1):
        new_rnn_input = np.load('./data/' + sys.argv[1] + '/rnn_input_' + str(batch_num) + '.npy')
        new_rnn_output = np.load('./data/' + sys.argv[1] + '/rnn_output_' + str(batch_num) + '.npy')

        if batch_num > config.START_BATCH:
            rnn_input = np.concatenate([rnn_input, new_rnn_input])
            rnn_output = np.concatenate([rnn_output, new_rnn_output])
        else:
            rnn_input = new_rnn_input
            rnn_output = new_rnn_output
    rnn.train(rnn_input, rnn_output)

if __name__ == "__main__":
    main()
# python 02_train_vae.py --new_model

import sys
import numpy as np

from keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Lambda, Reshape, BatchNormalization
from keras.models import Model
from keras import backend as K
from keras.callbacks import EarlyStopping

import config

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], config.Z_DIM), mean=0., stddev=1.)
    return z_mean + K.exp(z_log_var / 2) * epsilon

class VAE():
    def __init__(self):
        self.models = self._build()
        self.model = self.models[0]
        self.encoder = self.models[1]
        self.decoder = self.models[2]

        self.input_dim = (64, 64, 3)

    def _build(self):
        input_x = Input(shape=(64, 64, 3))

        conv1 = Conv2D(filters=32, kernel_size=4, strides=2,
                        activation='relu')(input_x)

        conv2 = Conv2D(filters=64, kernel_size=4, strides=2,
                        activation='relu')(conv1)

        conv3 = Conv2D(filters=64, kernel_size=4, strides=2,
                        activation='relu')(conv2)

        conv4 = Conv2D(filters=128, kernel_size=4, strides=2,
                        activation='relu')(conv3)


        z_in = Flatten()(conv4)

        z_mean = Dense(config.Z_DIM)(z_in)
        z_log_var = Dense(config.Z_DIM)(z_in)

        z = Lambda(sampling)([z_mean, z_log_var])
        z_input = Input(shape=(config.Z_DIM,))

        # These layers are used later on.
        dense1 = Dense(1024)
        dense_model = dense1(z)

        z_out = Reshape((1, 1, 1024))
        z_out_model = z_out(dense_model)

        d1 = Conv2DTranspose(filters=64, kernel_size=5,
                                 strides=2, activation='relu')
        d1_model = d1(z_out_model)

        d2 = Conv2DTranspose(filters=64, kernel_size=5,
                                 strides=2, activation='relu')
        d2_model = d2(d1_model)

        d3 = Conv2DTranspose(filters=32, kernel_size=6,
                                 strides=2, activation='relu')
        d3_model = d3(d2_model)

        d4 = Conv2DTranspose(filters=3, kernel_size=6,
                                 strides=2, activation='sigmoid')
        d4_model = d4(d3_model)


        # Decoder
        dense_decoder = dense1(z_input)
        z_out_decoder = z_out(dense_decoder)

        d1_decoder = d1(z_out_decoder)
        d2_decoder = d2(d1_decoder)
        d3_decoder = d3(d2_decoder)
        d4_decoder = d4(d3_decoder)

        # Create encoder and decoder models
        vae = Model(input_x, d4_model)
        vae_encoder = Model(input_x, z)
        vae_decoder = Model(z_input, d4_decoder)

        def r_loss(y_true, y_pred):
            y_true_flat = K.flatten(y_true)
            y_pred_flat = K.flatten(y_pred)

            return 10 * 255 * K.mean(K.square(y_true_flat - y_pred_flat), axis=-1)

        def kl_loss(y_true, y_pred):
            return - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)

        def loss_func(y_true, y_pred):
            return r_loss(y_true, y_pred) + kl_loss(y_true, y_pred)

        vae.compile(optimizer='rmsprop', loss=loss_func)  # ,  metrics = [vae_r_loss, vae_kl_loss])
        return (vae, vae_encoder, vae_decoder)

    def set_weights(self, filepath):
        self.model.load_weights(filepath)

    def train(self, data, validation_split=0.2):
        print('data shape: {}'.format(data.shape))
        earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5, verbose=1, mode='auto')
        callbacks_list = [earlystop]
        self.model.fit(data, data,
                       shuffle=True,
                       epochs=config.EPOCHS_VAE,
                       batch_size=config.BATCH_SIZE_VAE,
                       validation_split=validation_split,
                       callbacks=callbacks_list)
        self.model.save_weights('./weights/vae/weights_' + sys.argv[1] +'.h5')

    def save_weights(self, filepath):
        self.model.save_weights(filepath)

    def generate_rnn_data(self, obs_data, action_data):
        rnn_input = []
        rnn_output = []

        for i, j in zip(obs_data, action_data):
            z_input = self.encoder.predict(np.array(i))
            conc = [np.concatenate([x, [y]]) for x, y in zip(z_input, j)]
            rnn_input.append(conc[:-1])
            rnn_output.append(np.array(z_input[1:]))

        rnn_input = np.array(rnn_input)
        rnn_output = np.array(rnn_output)

        return (rnn_input, rnn_output)

def main():
    vae = VAE()

    if not config.NEW_MODEL:
        vae.set_weights('./weights/vae/weights.h5')

    for batch_num in range(config.START_BATCH, config.MAX_BATCH + 1):
        print('Creating batch {}...'.format(batch_num))
        first_iter = True

        new_data = np.load('data/' + sys.argv[1] +'/obs_data_' + config.ENV_NAME + '_' + str(batch_num) + '.npy')

        if first_iter:
            data = new_data
            first_iter = False
        else:
            data = np.concatenate([data, new_data])
        print('Found {}...current data size = {} episodes'.format(config.ENV_NAME, len(data)))

        if first_iter == False:  # i.e. data has been found for this batch number
            data = np.array([item for obs in data for item in obs])
            vae.train(data)
        else:
            print('no data found for batch number {}'.format(batch_num))

if __name__ == "__main__":
    main()
    # Second argument use 1 if you want to generate images using the VAE after training.
    if len(sys.argv) > 2:
        if int(sys.argv[2]) == 1:
            import model
            amount_to_sample = 30
            vae = VAE()
            vae.set_weights('./weights/vae/weights_{}.h5'.format(sys.argv[1]))
            obs = np.load('data/' + sys.argv[1] +'/obs_data_' + config.ENV_NAME + '_' + str(0) + '.npy')
            counter = 0
            for i in range(amount_to_sample):
                random_pick = np.random.randint(0, len(obs))
                o = obs[random_pick].squeeze()
                encoded_obs = vae.encoder.predict(np.array(o))[random_pick]
                model.save_deconstruct_vae_img('./images/decoded_VAE/', vae, encoded_obs, random_pick, counter, original=o[random_pick], show_img=False)
                counter+=1

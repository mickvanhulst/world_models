# python model.py car_racing --filename ./controller/car_racing.cma.4.32.best.json --render_mode --record_video
# xvfb-run -a -s "-screen 0 1400x900x24" python model.py car_racing --filename ./controller/car_racing.cma.4.32.best.json --render_mode --record_video
import copy
import sys
import numpy as np
import random
import json
import time
import argparse
import config
import os

from train_vae import VAE
from train_rnn import RNN
import matplotlib.pyplot as plt
import tensorflow as tf
import math

from skimage.transform import resize

final_mode = False
render_mode = False
generate_data_mode = False
RENDER_DELAY = False
record_video = False
MEAN_MODE = False

def activations(a):
  a = np.tanh(a)
  return a

class Controller():
    def __init__(self):
        self.time_factor = config.TIME_FACTOR
        self.noise_bias = config.NOISE_BIAS
        self.output_noise=config.OUTPUT_NOISE
        self.activations=activations
        self.output_size = config.CONTR_OUTPUT_SIZE

def make_model(experiment_name, gen_vae_data=False):
    vae = VAE()
    vae.set_weights('./weights/vae/weights_{}.h5'.format(experiment_name))

    rnn = RNN()
    rnn.set_weights('./weights/rnn/weights_{}.h5'.format(experiment_name))

    controller = Controller()

    model = Model(controller, vae, rnn)
    if gen_vae_data:
        return model, vae
    else:
        return model


class Model:
    def __init__(self, controller, vae, rnn):

        self.input_size = vae.input_dim
        self.vae = vae
        self.rnn = rnn

        self.output_noise = controller.output_noise
        self.sigma_bias = controller.noise_bias  # bias in stdev of output
        self.sigma_factor = 0.5  # multiplicative in stdev of output

        if controller.time_factor > 0:
            self.time_factor = float(controller.time_factor)
            self.time_input = 1
        else:
            self.time_input = 0

        self.output_size = controller.output_size

        self.sample_output = False
        self.activations = controller.activations

        self.weight = []
        self.bias = []
        self.bias_log_std = []
        self.bias_std = []
        self.param_count = 0

        self.hidden = np.zeros(self.rnn.hidden_units)
        self.cell_values = np.zeros(self.rnn.hidden_units)

        self.shapes = [(self.rnn.hidden_units + config.Z_DIM, self.output_size)]

        idx = 0
        for shape in self.shapes:
            self.weight.append(np.zeros(shape=shape))
            self.bias.append(np.zeros(shape=shape[1]))
            self.param_count += (np.product(shape) + shape[1])
            if self.output_noise[idx]:
                self.param_count += shape[1]
            log_std = np.zeros(shape=shape[1])
            self.bias_log_std.append(log_std)
            out_std = np.exp(self.sigma_factor * log_std + self.sigma_bias)
            self.bias_std.append(out_std)
            idx += 1

        self.render_mode = False

    def make_env(self, seed=111):
        self.render_mode = render_mode
        self.env = config.make_env(seed=seed)

    def get_action(self, x, t=0, mean_mode=False):
        # if mean_mode = True, ignore sampling.
        h = np.array(x).flatten()
        if self.time_input == 1:
            time_signal = float(t) / self.time_factor
            h = np.concatenate([h, [time_signal]])
        num_layers = len(self.weight)
        for i in range(num_layers):
            w = self.weight[i]
            b = self.bias[i]
            h = np.matmul(h, w) + b
            if (self.output_noise[i] and (not mean_mode)):
                out_size = self.shapes[i][1]
                out_std = self.bias_std[i]
                output_noise = np.random.randn(out_size) * out_std
                h += output_noise

            h = self.activations(h)

        return h

    def set_model_params(self, model_params):
        pointer = 0
        for i in range(len(self.shapes)):
            w_shape = self.shapes[i]
            b_shape = self.shapes[i][1]
            s_w = np.product(w_shape)
            s = s_w + b_shape
            chunk = np.array(model_params[pointer:pointer + s])
            self.weight[i] = chunk[:s_w].reshape(w_shape)
            self.bias[i] = chunk[s_w:].reshape(b_shape)
            pointer += s
            if self.output_noise[i]:
                s = b_shape
                self.bias_log_std[i] = np.array(model_params[pointer:pointer + s])
                self.bias_std[i] = np.exp(self.sigma_factor * self.bias_log_std[i] + self.sigma_bias)
                if self.render_mode:
                    print("bias_std, layer", i, self.bias_std[i])
                pointer += s

    def load_model(self, filename):
        with open(filename) as f:
            data = json.load(f)
        print('loading file %s' % (filename))
        self.data = data
        model_params = np.array(data[0])  # assuming other stuff is in data
        self.set_model_params(model_params)

    def get_random_model_params(self, stdev=0.1):
        return np.random.randn(self.param_count) * stdev

    def reset(self):
        self.hidden = np.zeros(self.rnn.hidden_units)
        self.cell_values = np.zeros(self.rnn.hidden_units)

    def update(self, obs, t):
        vae_encoded_obs = self.vae.encoder.predict(np.array([obs]))[0]
        return vae_encoded_obs


def evaluate(model):
    # run 100 times and average score, according to the reles.
    model.env.seed(0)
    total_reward = 0.0
    N = 100
    for i in range(N):
        reward, t = simulate(model, train_mode=False, render_mode=False, num_episode=1)
        print("Reward: ", reward, "Current total reward: ", total_reward)
        total_reward += reward[0]
    return (total_reward / float(N))

def save_deconstruct_vae_img(path, vae, obs, t, counter, original=None, show_img=False):
    # Decode VAE obs and reconstruct
    vae_decoded_obs = vae.decoder.predict(np.array([obs]))
    vae_decoded_obs *= 255
    plt.imsave('{}{}/timestep_{}_{}.png'.format(path, config.Z_DIM, t, counter),
               vae_decoded_obs.astype(int).squeeze())
    if original is not None:
        print('image shape {}'.format(original.shape))
        original *= 255
        plt.imsave('{}{}/timestep_{}_{}_original.png'.format(path, config.Z_DIM, t, counter),
                   original.astype(int).squeeze())
    if show_img:
        plt.imshow(vae_decoded_obs.astype(int).squeeze())
        plt.show()

def get_mixture_coef_cust(y_pred):
    d = config.GAUSSIAN_MIXTURES * config.Z_DIM

    rollout_length = np.shape(y_pred)[1]

    pi = y_pred[:, :, :d]
    mu = y_pred[:, :, d:(2 * d)]
    log_sigma = y_pred[:, :, (2 * d):(3 * d)]

    pi = np.reshape(pi, [-1, rollout_length, config.GAUSSIAN_MIXTURES, config.Z_DIM])
    mu = np.reshape(mu, [-1, rollout_length, config.GAUSSIAN_MIXTURES, config.Z_DIM])
    log_sigma = np.reshape(log_sigma, [-1, rollout_length, config.GAUSSIAN_MIXTURES, config.Z_DIM])

    pi = np.exp(pi) / np.sum(np.exp(pi), axis=2, keepdims=True)
    sigma = np.exp(log_sigma)

    return pi, mu, sigma  # , discrete


def tf_normal_cust(y_true, mu, sigma, pi):
    rollout_length = np.shape(y_true)[1]
    y_true = np.tile(y_true, (1, 1, config.GAUSSIAN_MIXTURES))
    y_true = np.reshape(y_true, [-1, rollout_length, config.GAUSSIAN_MIXTURES, config.Z_DIM])

    oneDivSqrtTwoPI = 1 / math.sqrt(2 * math.pi)
    result = y_true - mu
    result = result * (1 / (sigma + 1e-8))
    result = -np.square(result) / 2
    result = np.exp(result) * (1 / (sigma + 1e-8)) * oneDivSqrtTwoPI
    result = result * pi
    result = np.sum(result, axis=2)
    return result

def simulate(model, train_mode=False, render_mode=True, num_episode=1, seed=-1, max_len=-1, generate_data_mode=False, save_images_VAE=False, counter=0):
    reward_list = []
    t_list = []

    max_episode_length = 3000

    if max_len > 0:
        if max_len < max_episode_length:
            max_episode_length = max_len

    if (seed >= 0):
        random.seed(seed)
        np.random.seed(seed)
        model.env.seed(seed)

    for e in range(num_episode):

        model.reset()
        obs = model.env.reset()
        obs = resize(obs, (64, 64, 3))

        action = model.env.action_space.sample()
        model.env.render("human")

        if obs is None:
            obs = np.zeros(model.input_size)

        total_reward = 0.0

        obs_sequence = []
        action_sequence = []
        for t in range(max_episode_length):

            if render_mode:
                model.env.render("human")
                if RENDER_DELAY:
                    time.sleep(0.01)

            obs_sequence.append(obs)
            action_sequence.append(action)

            vae_encoded_obs = model.update(obs, t)

            controller_obs = np.concatenate([vae_encoded_obs, model.hidden])
            action = model.get_action(controller_obs, t=t, mean_mode=False)
            action = np.argmax(action)

            obs, reward, done, info = model.env.step(action)
            obs = resize(obs, (64, 64, 3))

            input_to_rnn = [np.array([[np.concatenate([vae_encoded_obs, [action]])]]), np.array([model.hidden]),
                            np.array([model.cell_values])]

            mdn, h, c = model.rnn.forward.predict(input_to_rnn)
            if save_images_VAE:
                if t % 25 == 0:
                    save_deconstruct_vae_img('./images/decoded_VAE/', vae, vae_encoded_obs, t, counter)

                    pi, mu, sigma = get_mixture_coef_cust(mdn)
                    result = tf_normal_cust(mdn, pi, mu, sigma)

                    for z in range(15):
                        save_deconstruct_vae_img('./images/decoded_RNN/', vae, result[z][0], t + z, counter)

            model.hidden = h[0]
            model.cell_values = c[0]

            total_reward += reward

            if done:
                break

        if render_mode:
            print("reward", total_reward, "timesteps", t)

        t_list.append(t)
        reward_list.append(total_reward)

    # Save data if saving data using model.py.
    if generate_data_mode:
        return reward_list, t_list, obs_sequence, action_sequence
    else:
        model.env.close()

    return reward_list, t_list

def main(file_name, generate_data_mode, render_mode, find_custom_steps, perc, pick_custom_steps_after_min, save_images_VAE, gen_int=True):
    env_name = 'space'
    filename = file_name
    the_seed = 111

    model, vae_res = make_model(sys.argv[1], gen_vae_data=True)
    global vae
    vae = vae_res
    model.make_env()

    if len(filename) > 0:
        model.load_model(filename)
    else:
        params = model.get_random_model_params(stdev=0.1)
        model.set_model_params(params)

    # if final_mode:
    total_reward = 0.0
    np.random.seed(the_seed)
    model.env.seed(the_seed)

    # Init params
    batch_count = 0
    s = 0
    episode_length = config.MAX_LENGTH
    shape_list = []
    cnt = 0
    while s < config.TOTAL_EPS:
        print(s, " out of 15 collected")
        obs_data = []
        action_data = []
        batch_size = min(config.BATCH_SIZE_GEN_DATA, config.TOTAL_EPS)

        if batch_size <= 0:
            batch_size = 1

        i = 0
        while i < batch_size:
            reward, steps_taken, obs_sequence, action_sequence = simulate(model, train_mode=False, render_mode=render_mode, num_episode=1,
                                           max_len=config.MAX_LENGTH, generate_data_mode=generate_data_mode, save_images_VAE=save_images_VAE, counter=cnt)
            cnt +=1
            total_reward += reward[0]
            print("episode", i, "reward =", reward[0])
            obs_sequence = np.array(obs_sequence)

            # Only save if minimum episode length of MAX_LENGTH timesteps.
            if not find_custom_steps:
                if gen_int:
                    rand_start = np.random.randint(5, 75)
                    gen_int = False
                    print(episode_length + rand_start)
                if obs_sequence.shape[0] >= (episode_length + rand_start):
                    obs_data.append(obs_sequence[rand_start:episode_length + rand_start,:,:])
                    action_data.append(action_sequence)
                    s += 1
                    i += 1
                    gen_int = True
            else:
                shape_list.append(obs_sequence.shape[0])
                i += 1
                if i == min(pick_custom_steps_after_min, batch_size - 1):
                    episode_length = int(np.percentile(shape_list, perc))
                    print('After {} iterations we picked the value {} as a custom step'.format(i, episode_length))
                    i = 0
                    find_custom_steps = False
                    del shape_list

        # Create folder if not exists
        save_directory = 'data/' + sys.argv[1]
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        np.save(save_directory + '/obs_data_' + config.ENV_NAME + '_' + str(batch_count), obs_data)
        np.save(save_directory + '/action_data_' + config.ENV_NAME + '_' + str(batch_count), action_data)

        print("seed", the_seed, "average_reward", total_reward / 100)

        batch_count += 1

    model.env.close()

if __name__ == "__main__":
    file_name = './weights/controller/space.cma.' + sys.argv[1] +'.best.json'
    generate_data_mode = True
    render_mode = True
    # Set this to False if you just want to use the settings from config.py
    find_custom_steps = True
    perc = 50
    save_images_VAE = True
    # Custom steps after we pick our 60th percentile, min(batch_size, pick_custom..).
    pick_custom_steps_after_min = 8
    main(file_name, generate_data_mode, render_mode, find_custom_steps, perc, pick_custom_steps_after_min, save_images_VAE)

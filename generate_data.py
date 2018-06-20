# xvfb-run -s "-screen 0 1400x900x24" python generate_data.py car_racing --total_episodes 200 --start_batch 0 --time_steps 300

import sys

import numpy as np
import random
import config
from skimage.transform import resize
from train_vae import VAE

from config import make_env

def gen_rand_data():
    env = make_env()
    s = 0
    batch = config.START_BATCH

    batch_size = min(config.BATCH_SIZE_GEN_DATA, config.TOTAL_EPS)
    while s < config.TOTAL_EPS:
        obs_data = []
        action_data = []

        for i_episode in range(batch_size):
            print('-----')
            observation = env.reset()
            observation = resize(observation, (64, 64, 3))

            if config.RENDER:
                env.render()
            t = 0
            obs_sequence = []
            action_sequence = []

            while t < config.TIME_STEPS:
                t = t + 1
                action = env.action_space.sample()
                obs_sequence.append(observation)
                action_sequence.append(action)

                observation, reward, done, info = env.step(action)
                observation = resize(observation, (64, 64, 3))

                if config.RENDER:
                    env.render()

            print('Shape for network: {}'.format(np.array(obs_data).shape))
            obs_data.append(obs_sequence)
            action_data.append(action_sequence)

            print("Batch {} Episode {} finished after {} timesteps".format(batch, i_episode, t + 1))
            print("Current dataset contains {} observations".format(sum(map(len, obs_data))))

            s += 1

        print("Saving dataset for batch {}".format(batch))
        np.save('data/' + sys.argv[1] + '/obs_data_' + config.ENV_NAME + '_' + str(batch), obs_data)
        np.save('data/' + sys.argv[1] + '/action_data_' + config.ENV_NAME + '_' + str(batch), action_data)

        batch = batch + 1

    env.close()

def gen_rnn_data():
    vae = VAE()
    vae.set_weights('./weights/vae/weights_' + sys.argv[1] +'.h5')

    for batch_num in range(config.START_BATCH, config.MAX_BATCH + 1):
        first_iter = True
        print('Generating batch {}...'.format(batch_num))

        new_obs_data = np.load('data/' + sys.argv[1] + '/obs_data_' + config.ENV_NAME + '_' + str(batch_num) + '.npy')
        new_action_data = np.load('data/'+ sys.argv[1] + '/action_data_' + config.ENV_NAME + '_' + str(batch_num) + '.npy')

        if first_iter:
            obs_data = new_obs_data
            action_data = new_action_data
            first_iter = False
        else:
            obs_data = np.concatenate([obs_data, new_obs_data])
            action_data = np.concatenate([action_data, new_action_data])
        print('Found {}...current data size = {} episodes'.format(config.ENV_NAME, len(obs_data)))

        if first_iter == False:
            rnn_input, rnn_output = vae.generate_rnn_data(obs_data, action_data)
            np.save('./data/' + sys.argv[1] + '/rnn_input_' + str(batch_num), rnn_input)
            np.save('./data/' + sys.argv[1] + '/rnn_output_' + str(batch_num), rnn_output)
        else:
            print('no data for batch number {}'.format(batch_num))


if __name__ == "__main__":
    if sys.argv[2] == 'rnn':
        gen_rnn_data()
    elif sys.argv[2] == 'rand':
        gen_rand_data()
import gym

# Constants used throughout entire application.
START_BATCH = 0
MAX_BATCH = 1
TOTAL_EPS = 64
TIME_STEPS = 320
RENDER = True
ENV_NAME = 'space'
NEW_MODEL = True
Z_DIM = 32
ACTION_DIM = 1

# Generate data constants
BATCH_SIZE_GEN_DATA = 32

# VAE constants
EPOCHS_VAE = 10
BATCH_SIZE_VAE = 32

# RNN constants
GAUSSIAN_MIXTURES = 5
RNN_BATCH_SIZE = 32
RNN_EPOCHS = 20

# Controller constants
# num_worker  : set this to no more than number of cores available
# num_work_trial : the number of members of the population that each worker will test (num_worker * num_work_trial gives the total population size for each generation)
# num_episode 4 : the number of episodes each member of the population will be scored against (i.e. the score will be the average reward across this number of episodes)
# max_length 1000 : the maximum number of time-steps in an episode
# eval_steps 25: the number of generations between the evaluation of the best set of weights, across 100 episodes
OPTIMIZER = 'cma'
INIT_OPT = ''
NUM_EPISODE = 1
NUM_WORKER = 2
NUM_WORKER_TRIAL = 1
EVAL_STEPS = 1
MAX_LENGTH = 2500
ANTITHETIC = 1
CAP_TIME = 0
RETRAIN = 0
SEED_START = 4711
SIGMA_INIT = 0.50
SIGMA_DECAY = 0.999
POPULATION = NUM_WORKER * NUM_WORKER_TRIAL
BATCH_MODE = 'mean'
FILEBASE = './log/' + ENV_NAME + '.' + OPTIMIZER + '.' + str(NUM_EPISODE) + '.' + str(POPULATION)
CONTROLLER_FILEBASE = './weights/controller/' + ENV_NAME + '.' + OPTIMIZER + '.'

# Model constants
TIME_FACTOR = 0
NOISE_BIAS = 0
OUTPUT_NOISE = [False, False, False]
CONTR_OUTPUT_SIZE = 6

def make_env(seed=111):
    GAME = ['Assault-v0', 'SpaceInvaders-v0']
    env = gym.make(GAME[1])
    print(seed)
    if (seed >= 0):
        env.seed(seed)

    return env
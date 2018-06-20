# python 05_train_controller.py car_racing -e 1 -n 4 -t 1 --max_length 1000
# xvfb-run -a -s "-screen 0 1400x900x24" python 05_train_controller.py car_racing -n 16 -t 2 -e 4 --max_length 1000

from mpi4py import MPI
import numpy as np
import json
import os
import subprocess
import sys
import pandas as pd

import pickle
from model import make_model, simulate
import time

import config
from mpi4py_agent import initialize_settings, OldSeeder, Seeder, encode_solution_packets, decode_solution_packet, encode_result_packet, \
    decode_result_packet, worker, send_packets_to_slaves, receive_packets_from_slaves, evaluate_batch

### MPI NEEDS to be here.
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def sprint(*args):
    print(args)  # if python3, can do print(*args)
    sys.stdout.flush()

def slave():
    new_model = make_model(sys.argv[1])
    while True:
        packet = comm.recv(source=0)
        packet = packet['result']

        solutions = decode_solution_packet(packet)
        results = []

        new_model.make_env()

        for solution in solutions:
            worker_id, jobidx, seed, train_mode, max_len, weights = solution

            worker_id = int(worker_id)
            jobidx = int(jobidx)
            seed = int(seed)

            fitness, timesteps = worker(weights, seed, max_len, new_model, train_mode)

            results.append([worker_id, jobidx, fitness, timesteps])

        new_model.env.close()

        result_packet = encode_result_packet(results)
        comm.Send(result_packet, dest=0)

def master():
    start_time = int(time.time())

    individual_stats = []
    population_stats = []
    mean_dist_stats = []    

    sys.stdout.flush()

    seeder = Seeder(config.SEED_START)
    filename_best = config.CONTROLLER_FILEBASE  +  sys.argv[1] + '.best.json'

    t = 0

    model.make_env()

    history = []
    eval_log = []
    best_reward_eval = 0
    best_model_params_eval = None

    while True:
        t += 1

        solutions = es.ask()

        if config.ANTITHETIC:
            seeds = seeder.next_batch(int(es.popsize / 2))
            seeds = seeds + seeds
        else:
            seeds = seeder.next_batch(es.popsize)

        packet_list = encode_solution_packets(seeds, solutions, max_len=config.MAX_LENGTH)

        reward_list = np.zeros(config.POPULATION)
        time_list = np.zeros(config.POPULATION)

        send_packets_to_slaves(packet_list, config.ENV_NAME)
        packets_from_slaves = receive_packets_from_slaves()
        reward_list = reward_list + packets_from_slaves[:, 0]
        time_list = time_list + packets_from_slaves[:, 1]

        mean_time_step = int(np.mean(time_list) * 100) / 100.  # get average time step
        max_time_step = int(np.max(time_list) * 100) / 100.  # get max time step
        avg_reward = int(np.mean(reward_list) * 100) / 100.  # get average reward
        std_reward = int(np.std(reward_list) * 100) / 100.  # get std reward

        es.tell(reward_list)

        es_solution = es.result()
        model_params = es_solution[0]
        model.set_model_params(np.array(model_params).round(4))

        r_max = int(np.max(reward_list) * 100) / 100.
        r_min = int(np.min(reward_list) * 100) / 100.

        curr_time = int(time.time()) - start_time
        
        individual_stats.append(reward_list)      
 
        population_stats.append([avg_reward, r_min, r_max, std_reward, int(es.rms_stdev() * 100000) / 100000., mean_time_step + 1.])

        h = (
        t, curr_time, avg_reward, r_min, r_max, std_reward, int(es.rms_stdev() * 100000) / 100000., mean_time_step + 1.,
        int(max_time_step) + 1)

        history.append(h)

        if (t == 1):
            best_reward_eval = avg_reward

        # Evaluate after EVAL_STEPS and save parameters if necessary.
        if (t % config.EVAL_STEPS == 0):

            prev_best_reward_eval = best_reward_eval
            model_params_quantized = np.array(es.current_param()).round(4)
            reward_eval = evaluate_batch(model_params_quantized, max_len=-1)
            print("Current reward during 100-fold evaluation: ", reward_eval)
            model_params_quantized = model_params_quantized.tolist()
            improvement = reward_eval - best_reward_eval
            eval_log.append([t, reward_eval, model_params_quantized])
            if (len(eval_log) == 1 or reward_eval > best_reward_eval):
                best_reward_eval = reward_eval
                best_model_params_eval = model_params_quantized
            else:
                if config.RETRAIN:
                    sprint("reset to previous best params, where best_reward_eval =", best_reward_eval)
                    es.set_mu(best_model_params_eval)
            with open(filename_best, 'wt') as out:
                res = json.dump([best_model_params_eval, best_reward_eval], out, sort_keys=True, indent=0,
                                separators=(',', ': '))
            mean_dist_stats.append([improvement, reward_eval, prev_best_reward_eval])
            sprint("improvement", t, improvement, "curr", reward_eval, "prev", prev_best_reward_eval, "best",
                   best_reward_eval)

            pd.DataFrame(mean_dist_stats).to_csv('mean_dist_stats_' + sys.argv[1] +'.csv')

        pd.DataFrame(individual_stats).to_csv('individual_stats_' + sys.argv[1] + '.csv')
        pd.DataFrame(population_stats).to_csv('population_stats_' + sys.argv[1] +'.csv')


def main():
    global es, model, PRECISION, SOLUTION_PACKET_SIZE, RESULT_PACKET_SIZE
    es, model, PRECISION, SOLUTION_PACKET_SIZE, RESULT_PACKET_SIZE = initialize_settings(comm, rank, config.SIGMA_INIT,
                                                                                         config.SIGMA_DECAY, config.INIT_OPT)

    if (rank == 0):
        master()
    else:
        slave()

def mpi_fork(n):
    """Re-launches the current script with workers
    Returns "parent" for original parent, "child" for MPI children
    (from https://github.com/garymcintire/mpi_util/)
    """
    if n <= 1:
        return "child"
    if os.getenv("IN_MPI") is None:
        env = os.environ.copy()
        env.update(
            MKL_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            IN_MPI="1"
        )
        print(["mpirun", "-np", str(n), sys.executable] + sys.argv)
        subprocess.check_call(["mpirun", "-np", str(n), sys.executable] + ['-u'] + sys.argv, env=env)
        return "parent"
    else:
        global nworkers, rank
        nworkers = comm.Get_size()
        rank = comm.Get_rank()
        print('assigning the rank and nworkers', nworkers, rank)
        return "child"


if __name__ == "__main__":
    if "parent" == mpi_fork(config.NUM_WORKER + 1): os.exit()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    main()

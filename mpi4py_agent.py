from mpi4py import MPI
import numpy as np
import json
import os
import subprocess
import sys
import pandas as pd

import pickle
from model import make_model, simulate
from es import CMAES
import config

import time

def initialize_settings(c, r, sigma_init=0.1, sigma_decay=0.9999, init_opt=''):
    global es, model, comm, rank
    comm, rank = c, r
    model = make_model(sys.argv[1])

    num_params = model.param_count

    if len(init_opt) > 0:
        es = pickle.load(open(init_opt, 'rb'))
    else:
        if config.OPTIMIZER == 'cma':
            cma = CMAES(num_params,
                        sigma_init=sigma_init,
                        popsize=config.POPULATION)
            es = cma

    global PRECISION
    PRECISION = 10000
    global SOLUTION_PACKET_SIZE
    SOLUTION_PACKET_SIZE= (5 + num_params) * config.NUM_WORKER_TRIAL
    global RESULT_PACKET_SIZE
    RESULT_PACKET_SIZE = 4 * config.NUM_WORKER_TRIAL

    return es, model, PRECISION, SOLUTION_PACKET_SIZE, RESULT_PACKET_SIZE

class OldSeeder:
    def __init__(self, init_seed=0):
        self._seed = init_seed

    def next_seed(self):
        result = self._seed
        self._seed += 1
        return result

    def next_batch(self, batch_size):
        result = np.arange(self._seed, self._seed + batch_size).tolist()
        self._seed += batch_size
        return result


class Seeder:
    def __init__(self, init_seed=0):
        np.random.seed(init_seed)
        self.limit = np.int32(2 ** 31 - 1)

    def next_seed(self):
        result = np.random.randint(self.limit)
        return result

    def next_batch(self, batch_size):
        result = np.random.randint(self.limit, size=batch_size).tolist()
        return result


def encode_solution_packets(seeds, solutions, train_mode=1, max_len=-1):
    n = len(seeds)
    result = []
    worker_num = 0
    for i in range(n):
        worker_num = int(i / config.NUM_WORKER_TRIAL) + 1
        result.append([worker_num, i, seeds[i], train_mode, max_len])
        result.append(np.round(np.array(solutions[i]) * PRECISION, 0))

    result = np.concatenate(result).astype(np.int32)
    result = np.split(result, config.NUM_WORKER)

    return result


def decode_solution_packet(packet):
    packets = np.split(packet, config.NUM_WORKER_TRIAL)
    result = []
    for p in packets:
        result.append([p[0], p[1], p[2], p[3], p[4], p[5:].astype(np.float) / PRECISION])
    return result


def encode_result_packet(results):
    r = np.array(results)
    r[:, 2:4] *= PRECISION
    return r.flatten().astype(np.int32)


def decode_result_packet(packet):
    r = packet.reshape(config.NUM_WORKER_TRIAL, 4)
    workers = r[:, 0].tolist()
    jobs = r[:, 1].tolist()
    fits = r[:, 2].astype(np.float) / PRECISION
    fits = fits.tolist()
    times = r[:, 3].astype(np.float) / PRECISION
    times = times.tolist()
    result = []
    n = len(jobs)
    for i in range(n):
        result.append([workers[i], jobs[i], fits[i], times[i]])
    return result


def worker(weights, seed, max_len, new_model, train_mode_int=1):
    train_mode = (train_mode_int == 1)
    new_model.set_model_params(weights)

    reward_list, t_list = simulate(new_model,
                                   train_mode=train_mode, render_mode=True, num_episode=config.NUM_EPISODE, seed=seed,
                                   max_len=max_len)

    if config.BATCH_MODE == 'min':
        reward = np.min(reward_list)
    else:
        reward = np.mean(reward_list)
    t = np.mean(t_list)
    return reward, t


def send_packets_to_slaves(packet_list, current_env_name):
    num_worker = comm.Get_size()
    #assert len(packet_list) == num_worker - 1
    for i in range(1, num_worker):
        packet = packet_list[i - 1]
        #assert (len(packet) == SOLUTION_PACKET_SIZE)
        packet = {'result': packet, 'current_env_name': current_env_name}
        comm.send(packet, dest=i)


def receive_packets_from_slaves():
    result_packet = np.empty(RESULT_PACKET_SIZE, dtype=np.int32)

    reward_list_total = np.zeros((config.POPULATION, 2))

    check_results = np.ones(config.POPULATION, dtype=np.int)
    for i in range(1, config.NUM_WORKER + 1):
        comm.Recv(result_packet, source=i)
        results = decode_result_packet(result_packet)
        for result in results:
            worker_id = int(result[0])
            possible_error = "work_id = " + str(worker_id) + " source = " + str(i)
            #assert worker_id == i, possible_error
            idx = int(result[1])
            reward_list_total[idx, 0] = result[2]
            reward_list_total[idx, 1] = result[3]
            check_results[idx] = 0

    check_sum = check_results.sum()
    #assert check_sum == 0, check_sum
    return reward_list_total


def evaluate_batch(model_params, max_len):
    # duplicate model_params
    solutions = []
    for i in range(es.popsize):
        solutions.append(np.copy(model_params))

    seeds = np.arange(es.popsize)

    packet_list = encode_solution_packets(seeds, solutions, train_mode=0, max_len=max_len)

    overall_rewards = []

    send_packets_to_slaves(packet_list, config.ENV_NAME)
    packets_from_slaves = receive_packets_from_slaves()
    reward_list = packets_from_slaves[:, 0] # get rewards
    print(reward_list)
    overall_rewards.append(np.mean(reward_list))
    print(overall_rewards)
    return np.mean(overall_rewards)
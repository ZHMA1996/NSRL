from core.argparser import *
from core.parallel_agent import *
from core.parallel_trainer import *
from core.parallel_env import *
import torch.optim as optim
from tianshou.utils import tqdm_config
import random
import gym
import tqdm
import time
import pickle
import numpy as np
from runx.logx import logx

import tianshou as ts
from tianshou.env import SubprocVectorEnv
from tianshou.data import ReplayBuffer, PrioritizedReplayBuffer
from tensorboardX import SummaryWriter
from tianshou.trainer.utils import test_episode

block_encoding = {"floor": 0, "a":1, "b": 2, "c":3, "d":4, "e": 5, "f":6, "g":7}
idx2block = {}
for key, value in block_encoding.items():
    idx2block[value] = key

predicate = {"On": 0, "Top":1, "GoalOn": 2}
idx2predicate = {}
for key, value in predicate.items():
    idx2predicate[value] = key

test_times = 10

def create_env(task='Stack', variations='None', test=False):
    
    env_fns = {'Stack': Stack, 'Unstack': Unstack, 'On':On}
    init_states = {'Stack' : INI_STATE2, 'Unstack':INI_STATE, 'On':INI_STATE}
    
    init_state = init_states[task]
    env_fn = env_fns[task]

    if variations != 'None':
        return env_fn(init_state).vary(variations, test)
    else:
        return env_fn(init_state, test=test)

seed = 200
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

if 'pg' in args.logdir:
    metaAgent = create_nsrl(device=args.device, model_type=args.model_type)
else:
    metaAgent = create_nsrl_ppo(device=args.device, model_type=args.model_type)

metaAgent.load_weights(args.logdir)
metaAgent.eval()

# Initialize neural network and agent

def test(env, metaAgent, visualize_rules=False):
    
    metaAgent.eval()

    rew_list = []

    vis = False

    with tqdm.tqdm(total=test_times, desc='Test', **tqdm_config) as t:

        for episodeCount in range(test_times):

            test_rew = 0

            obs = env.reset()

            episodeSteps = 0

            done = False

            while not done:


                meta_batch = Batch(obs=[obs])

                with torch.no_grad():

                    res = metaAgent(meta_batch, mode='test')

                    act = res.act[0]

                    atoms = env.compute_atoms(act)

                    if visualize_rules and not vis:

                        print('-' * 80)

                        all_path, possible_path = visualize_logical_path(obs, res.path_attn, res.pre_attn_list, atoms)

                obs, rew, done, info = env.step(act)

                test_rew += rew

            rew_list.append(test_rew)

            t.update()

            vis = True
        
        print(np.mean(rew_list), np.std(rew_list))

        return rew_list

def print_path(path):
    tmp = path[:2]
    tmp.extend([idx2block[path[i]] if type(path[i]) != type('a') else path[i] for i in range(2, len(path))])
    return tmp

def print_path_list(path_list):
    tmp = [print_path(y) for y in path_list]
    return tmp

def visualize_logical_path(obs, path_attn, pre_attn_list, atoms):
    
    a,b = atoms.terms
    a = block_encoding[a]
    b = block_encoding[b]

    obs = obs[:-1, :]    

    obs = obs.reshape((args.predicate_num, args.arity_num, args.arity_num))

    # extract the relational path has the highest attention weights
    all_path = []

    path_attn = path_attn.reshape(-1)

    pre_attn_list = [pre_attn.reshape(-1) for pre_attn in pre_attn_list]

    last_path_list = []

    possible_path = []

    tong_path = []

    path_len = args.path_length

    #path_len = path_attn.reshape(-1).argmax()+1
    path_len = args.path_length

    path_tmp = []

    for i in range(path_len):
        path_tmp = []
        index = np.argsort(pre_attn_list[i])
        if len(last_path_list):
            path_tmp = []
            for ind in index:
                for path in last_path_list:
                    coord = np.where(obs[ind, :, :] == 1)
                    if len(coord):
                        for j, k in zip(coord[0], coord[1]):
                            tmp = copy.deepcopy(path)
                            tmp[0] *= pre_attn_list[i][ind]
                            tmp.extend([j, k, idx2predicate[ind]])
                            if j != tmp[-5]:
                                tmp[1] = 0
                            if tmp[1] and k == b:
                                tmp[1] = 2
                                possible_path.append(tmp)
                            path_tmp.append(tmp)
        else:
            for ind in index:
                coord = np.where(obs[ind, :, :] == 1)
                if len(coord):
                    for j,k in zip(coord[0], coord[1]):
                        path_tmp.append([pre_attn_list[i][ind], j == a ,j, k, idx2predicate[ind]])
                        pass

        last_path_list = path_tmp
        #single_path_list = sorted(single_path_list, key=lambda x:x[0])
        all_path.append(path_tmp)

    for i in range(path_len):
        weights = []
        path_every_len = all_path[i]
        weights = [path[0] for path in path_every_len]
        weights = np.array(weights)
        weights = np.exp(weights)
        weights = weights / np.sum(weights)
        weights = weights.reshape((-1))
        weights *= path_attn[i]

        for ind, path in enumerate(path_every_len):
            path[0] = weights[ind]

    '''
    for path_every_len in all_path:
        final_path = print_path_list(path_every_len)
        for path in final_path:
            print(path)
    '''

    import pdb

    x = []
    for path_every_len in all_path:
        final_path = print_path_list(path_every_len)
        for path in final_path:
            if path[-2] == idx2block[b] and path[2] == idx2block[a] and path[1]:
                x.append(path)

    x = sorted(x, key=lambda y:-y[0])
    for path in x:
        print(path)
    
    return all_path, possible_path

def visualize_symbolic_logic_constrain_goal_linear(meta_obs, path_attn, pre_attn_list, goal, transition_matrix):
    
    import copy

    def print_path(path):

        tmp = [idx2loc[path[i]] if i % 2 != 0 and i != 0 else path[i] for i in range(len(path))]

        #tmp = [[idx2loc[x[0]],  idx2loc[x[1]], x[2]] for x in path]
        return tmp

    def print_path_list(path_list):
        tmp = [print_path(y) for y in path_list]
        return tmp

    meta_obs = meta_obs.reshape((13, 10, 10))

    # extract the relational path has the highest attention weights
    all_path = []
    path_attn = path_attn.reshape(-1)
    pre_attn_list = [pre_attn.reshape(-1) for pre_attn in pre_attn_list]

    last_path_list = []
    success_key_path = []

    #path_len = path_attn.reshape(-1).argmax()+1
    path_len = 8
    path_tmp = []


    for i in range(path_len):
        path_tmp = []
        index = 0
        while(len(path_tmp) == 0 and index >= -7):
            index -= 1
            ind = np.argsort(pre_attn_list[i])[index]
            if len(last_path_list):
                for path in last_path_list:
                    coord = np.where(1 == meta_obs[ind,  path[-1], :])[0]
                    if len(coord):
                        for next_object in coord:
                            tmp = copy.deepcopy(path)
                            tmp[0] *= pre_attn_list[i][ind]
                            tmp.extend([idx2predicate[ind], next_object])
                            path_tmp.append(tmp)
                            if next_object == goal:
                                success_key_path.append(path_tmp)

            else:
                coord = np.where(1 == meta_obs[ind, 0, :])[0]
                if len(coord):
                    for next_object in coord:
                        path_tmp.append([pre_attn_list[i][ind], 0, idx2predicate[ind], next_object])
        
        single_path_list = []
        for path in path_tmp:
            single_path = copy.deepcopy(path)
            single_path[0] *= path_attn[i]
            single_path_list.append(single_path)

        path_tmp = sorted(path_tmp, key=lambda x: x[0])
        last_path_list = path_tmp
        single_path_list = sorted(single_path_list, key=lambda x:-x[0])
        all_path.append(single_path_list)

    x = []
    for path_every_len in all_path:
        final_path = print_path_list(path_every_len)
        for path in final_path:
            if path[-1] == idx2loc[goal]:
                x.append(path)
    x = sorted(x, key=lambda y:-y[0])
    for path in x:
        print(path)


if args.task == 'Stack':
    all_variations = ("None","swap right 2","2 columns", "5 blocks",
                        "6 blocks", "7 blocks")
elif args.task == 'On':
    all_variations = ("None","swap top 2","swap middle 2", "5 blocks",
                "6 blocks", "7 blocks")
else:
    all_variations = ("None", "swap top 2","2 columns", "5 blocks",
                    "6 blocks", "7 blocks")

test_data = {}

test_str = ''

start = time.time()

for variations in all_variations:

    test_rew = []

    env = create_env(args.task, variations, True)

    vis = True if variations == 'None' else False

    rew_list = test(env, metaAgent, vis)

    test_str += (variations + ' ' + str(np.mean(rew_list))) + ' +- ' + str(np.std(rew_list)) + '\n'

end = time.time()

print('time use {0} s'.format(end - start))

print(test_str)

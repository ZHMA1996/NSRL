from core.argparser import *
from core.parallel_agent import *
from core.parallel_env import *
from core.parallel_trainer import *
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

metaAgent = create_nsrl_ppo(device=args.device, model_type=args.model_type)

if args.load_model:
    metaAgent.load_weights(args.load_model)

if args.test:

    metaAgent.load_weights(args.logdir)
    metaAgent.eval()

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
    for variations in all_variations:
        
        block_n = 4
        
        if variations in ['swap right 2', "2 columns", "swap top 2", "swap middle 2"]:
            block_n = 4
        elif variations == "5 blocks":
            block_n = 5
        elif variations == "6 blocks":
            block_n = 6
        elif variations == "7 blocks":
            block_n = 7

        test_rew = []


        test_envs = SubprocVectorEnv([lambda : create_env(args.task, variations, True) for _ in range(args.num_process)])
        
        test_collector = ts.data.Collector(metaAgent, test_envs)
        res = test_episode(metaAgent, test_collector, test_fn=None, epoch=0, n_episode=args.test_times)

        if variations == 'None':
            variations = args.task
        
        print(variations, res)

        '''
        test_data[variations] = [np.mean(test_rew), np.std(test_rew)]

        test_str += variations + ": " + "[{0}, {1}]\n".format(str(np.mean(test_rew)), str(np.std(test_rew)))
        '''
else:

    logx.initialize(logdir=args.logdir, hparams=vars(args), tensorboard=False)

    cumulative_average_reward = 0
    train_envs = SubprocVectorEnv([lambda : create_env(args.task) for _ in range(args.num_process)], wait_num=20)
    test_envs = SubprocVectorEnv([lambda : create_env(args.task) for _ in range(50)])

    train_collector = ts.data.Collector(metaAgent, train_envs, ReplayBuffer(size=20000))
    test_collector = ts.data.Collector(metaAgent, test_envs)

    writer = SummaryWriter(args.logdir)

    def save_fn(policy):
        policy.save_weights()

    def stop_fn(rew):
        '''
        if rew >= 0.94:
            return True
        '''
        return False


    res = onpolicy_trainer(metaAgent, train_collector, test_collector,
                max_epoch=args.max_epoch, step_per_epoch=args.step_per_epoch, 
                collect_per_step=args.collect_per_step, repeat_per_collect=args.repeat_per_collect,
                episode_per_test=args.episode_per_test, batch_size=args.batch_size, 
                stop_fn=stop_fn, save_fn=save_fn, writer=None, verbose=True)




import time
import os
import shutil
import numpy as np
import tianshou as ts

import copy
import torch
import torch.nn as nn
import torch.optim as optim

from tianshou.data import Batch
from tianshou.data import to_torch_as
from tianshou.data import to_numpy
from tianshou.policy import DQNPolicy, PGPolicy
from tianshou.data import PrioritizedReplayBuffer, ReplayBuffer

from tianshou.utils import tqdm_config
from tianshou.env import BaseVectorEnv
from tianshou.env.utils import CloudpickleWrapper


from modules import *
from env import *

import gym
import random
from torch.multiprocessing import Process, Pipe
from typing import List, Tuple, Union, Optional, Callable, Any

from argParser import *

class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, t):
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)

minReward = -10
maxReward = 10

alpha = args.alpha
beta = args.beta

max_timesteps=1000000
prioritized_replay_eps=1e-6
prioritized_replay_beta_iters = max_timesteps*0.5

beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                               initial_p=args.beta,
                               final_p=1.0)

lr = args.lr
rho = 0.95
eps=1e-08

actionMap = [0, 1, 2, 3, 4, 5, 11, 12]
goal_to_train = [i for i in range(7)]
Num_subgoal = len(goal_to_train)
maxStepsPerEpisode = 500

class DDQNPolicy(DQNPolicy):

    def __init__(self, model, optim, goal, 
                mask = None,
                discount_factor = 0.99,
                estimation_step = 1,
                target_update_freq = 400,
                reward_normalization = False,
                batch_size = 128,
                random_play_steps = 5000,
                buffer_size = 10000,
                explorationSteps = 200000,
                device = 'cpu',
                initial_eps=1.,
                final_eps=0.02,
                **kwargs):

        super().__init__(model=model, optim=optim, discount_factor=discount_factor, 
                        estimation_step=estimation_step, 
                        target_update_freq=target_update_freq, 
                        reward_normalization=reward_normalization)
        
        self.goal = goal
        self.mask = mask

        self.randomPlay = True
        self.random_play_steps = random_play_steps
        
        self.set_eps(initial_eps)
        self.success_ratio = -1

        self.device=device
        self.batch_size = batch_size
        self.buffer_size = buffer_size

        self.buffer = PrioritizedReplayBuffer(size=buffer_size, alpha=alpha, beta=beta)

        #self.buffer = ReplayBuffer(size=buffer_size, ignore_obs_next=True)
        #self.buffer = ReplayBuffer(size=buffer_size, alpha=alpha, beta=beta, ignore_obs_next=True)
        self.exploration = LinearSchedule(schedule_timesteps = explorationSteps, final_p = final_eps, initial_p=initial_eps)

        self.reset_exploration = False

    def forward(self, batch, model='model', input='obs', eps=None, mode='train', **kwargs):
        
        model = getattr(self, model)
        obs = getattr(batch, input)
        obs_ = obs.obs if hasattr(obs, 'obs') else obs

        # visualize attention weights
        path_attn, pre_attn_list, q_temp = None, None, None

        if not isinstance(obs_, torch.Tensor):
            obs_ = torch.tensor(obs_).float().to(self.device)
        
        q, path_attn, pre_attn_list = model(obs_)

        act = to_numpy(q.max(dim=1)[1])

        if self.mask != None:
            q_ = to_numpy(q)
            q_[:,self.mask] = -np.inf
            act = q_.argmax(axis=1)

        if eps is None:
            eps = self.eps
        if not np.isclose(eps, 0):
            for i in range(len(q)):
                if random.random() < eps:
                    q_ = np.array([random.random() for _ in range(*q[i].shape)])
                    #q_ = np.random.rand(*q[i].shape)
                    if self.mask != None:
                        q_[self.mask] = -np.inf
                    act[i] = q_.argmax()

        return Batch(logits=q, act=act, path_attn=path_attn, pre_attn_list=pre_attn_list)

    def criticize(self, reachGoal, die, distanceReward, useSparseReward):
        reward = -0.1
        if reachGoal:
            reward += 10
            #reward += 50.0
        if die:
            reward -= 5
        if not useSparseReward:
            reward += distanceReward
        reward = np.minimum(reward, maxReward)
        reward = np.maximum(reward, minReward)
        return reward

    def add(self, obs, act, rew, done, obs_next):
        self.buffer.add(obs=obs, act=act, rew=rew, done=done, obs_next=obs_next)
    
    def anneal_eps(self, step_count, option_learned=False):
        if option_learned:
            self.set_eps(0)
        else:
            if step_count > self.random_play_steps:
                self.set_eps(self.exploration.value(step_count - self.random_play_steps))

    def update(self):
        batch_data, indice = self.buffer.sample(batch_size = self.batch_size)
        batch_data = self.process_fn(batch_data, self.buffer, indice)
        metric = self.learn(batch_data)
        #self.buffer.anneal_beta(beta_schedule.value(step_count))
        return metric['loss']
    
    def save_weights(self, success_ratio):
        path = args.logdir + 'policy_subgoal_' + self.goal + '.pth'
        if success_ratio >= self.success_ratio:
            save_dict = {}
            
            if self.goal == 'meta':
                save_dict['meta'] = self.model.state_dict()
                save_dict['meta_optim'] = self.optim.state_dict()
                save_dict['meta_eps'] = self.eps
            else:   
                save_dict['goal' + str(self.goal)] = self.model.state_dict()
                save_dict['goal' + str(self.goal) + '_optim'] = self.optim.state_dict()
                save_dict['eps' + str(self.goal)] = self.eps
            torch.save(save_dict, path)
            self.success_ratio = success_ratio

    def load_weights(self, logdir ,subgoal, best=False):

        if best:
            path = logdir + 'policy_subgoal_' + self.goal + '.pth'
            weights = torch.load(path, map_location=torch.device(self.device))
            if subgoal != 'meta':
                self.model.load_state_dict(weights['goal'+str(subgoal)])
                self.optim.load_state_dict(weights['goal'+str(subgoal) + '_optim'])
                self.set_eps(weights['eps'+str(subgoal)])
            else:
                self.model.load_state_dict(weights['meta'])
                self.optim.load_state_dict(weights['meta_optim'])
                self.set_eps(weights['meta_eps'])

            
            print('Load weight successfully for goal ' + subgoal)
        else:
            import glob
            filelist = glob.glob(logdir + '*')
            for file_name in filelist:
                if 'last_checkpoint' in file_name:
                    weights = torch.load(file_name, map_location=torch.device(self.device))
                    if self.goal == 'meta':
                        self.model.load_state_dict(weights['meta'])
                        self.optim.load_state_dict(weights['meta_optim'])
                        self.set_eps(weights['meta_eps'])
                        print('Load weight successfully for meta controller ')
                        break
                    else:
                        self.model.load_state_dict(weights['goal' + subgoal])
                        self.optim.load_state_dict(weights['goal{0}_optim'.format(subgoal)])
                        self.set_eps(weights['eps' + subgoal])
                        print('Load weight successfully for goal {0} '.format(subgoal))
                        break
        if self._target:
            self.sync_weight()
    
    def reset_exploration(self, eps, steps, option_t):

        self.random_play_steps = option_t

        self.exploration = LinearSchedule(schedule_timesteps = steps, initial_p=eps, final_p=0.05)
    
    def clear_memory(self):
        self.buffer.reset()

    def operate_buffer(self, option_learned):
        
        buffer = getattr(self, 'buffer', None)
        if option_learned:
            if buffer is not None:
                del self.buffer
                self.buffer = []
        else:
            if type(buffer) == type([1,2]):
                self.buffer = PrioritizedReplayBuffer(size=self.buffer_size, alpha=alpha, beta=beta)

    def operate_exploration(self, eps, timesteps):

        if not self.reset_exploration:
            self.exploration = LinearSchedule(schedule_timesteps = timesteps + 20000, final_p = 0.002, initial_p=eps)
            self.reset_exploration = True

    def lr_schedule(self):
        for p in self.optim.param_groups:
            p['lr'] *= 0.99


class MetaModel(nn.Module):

    def __init__(self, predicate_num, arity_num, steps, output_embedding_size, n_layers, n_head):
        super().__init__()

        self.predicate_transformer = Transformer(steps = steps,
                                                 d_input=(arity_num ** 2),
                                                 inner= predicate_num * (arity_num ** 2),
                                                 output_embedding=True,
                                                 output_embedding_size=output_embedding_size,
                                                 n_layers=n_layers,
                                                 n_head=n_head)
        
        self.path_transformer = Transformer(steps=1,
                                            d_input=output_embedding_size,
                                            inner= predicate_num * (arity_num ** 2),
                                            output_embedding=False,
                                            output_embedding_size=128,
                                            n_layers=n_layers,
                                            n_head=n_head)
        
        self.steps = steps
        #self.transition_matrix = nn.Linear(arity_num, arity_num)
        self.predicate_num = predicate_num
        self.arity_num = arity_num

        transition_matrix = torch.randn((arity_num,arity_num), requires_grad=True)

        self.transition_matrix = torch.nn.Parameter(transition_matrix)
        self.register_parameter('Transition', self.transition_matrix)
        
        
        #self.linear1 = nn.Linear(arity_num ** 2, 512)
        #self.linear2 = nn.Linear(512, arity_num)
        
        self.linear = nn.Linear(arity_num ** 2, 512)
        self.linear1 = nn.Linear(512, 256)
        self.linear2 = nn.Linear(256, arity_num)

        #self.linear1 = nn.Linear(arity_num ** 2, 512)
        #self.linear2 = nn.Linear(512, arity_num)

        #self.linear1 = nn.Linear(1, 128)
        #self.linear2 = nn.Linear(128, 1)

    def forward(self, query, mode='train'):

        pre_output_list, pre_output_embedding_list, \
            pre_sfm_attn_list, pre_attn_list = self.predicate_transformer(query)
        
        path_transformer_input = torch.cat(pre_output_embedding_list, 1)

        path_output_list, path_sfm_attn_list, path_attn_list = \
            self.path_transformer(path_transformer_input)
        
        path_list = []
        predicate_matrix_step_list = []
        batch_size = query.shape[0]
        for i in range(len(pre_attn_list)):
            attn = pre_attn_list[i]
            
            matrix_each_step = (query * attn).sum(1).view(batch_size, self.arity_num, self.arity_num)

            predicate_matrix_step_list.append(matrix_each_step)
            if len(path_list):
                path_list.append(torch.matmul(matrix_each_step, path_list[-1]))
            else:
                path_list.append(matrix_each_step)
        
        path_attn = path_attn_list[-1]

        for indx in range(len(path_list)):
            path_list[indx] = path_list[indx].view(batch_size, 1, self.arity_num, self.arity_num)

        path = torch.cat(path_list, dim=1).view(batch_size, self.steps , -1)
        final_path = (path_attn * path).sum(1).view(batch_size, -1)

        #score = self.transition_matrix(final_path)
        #score = self.transition_matrix * final_path
        #score = torch.matmul(final_path, self.transition_matrix)
        #score = torch.matmul(self.transition_matrix, final_path)

        score = F.relu(self.linear(final_path))
        score = F.relu(self.linear1(score))
        score = self.linear2(score)
        #score = self.linear1(final_path)
        #score = self.linear2(score)
        
        #score = self.linear1(final_path)
        #score = self.linear2(score)
        
        #score = score.view(batch_size, self.arity_num, self.arity_num)

        #score = F.relu(self.linear1(final_path))
        #score = self.linear2(score)

        '''
        path = []
        for i in range(batch_size):
            path.append(torch.diagonal(final_path[i, :, :]))
        path = torch.cat(path, dim=0)
        final_path = path.view(batch_size, self.arity_num, 1)
        score = F.relu(self.linear1(final_path))
        score = self.linear2(score)
        '''
        
        return score, path_attn, pre_attn_list

class Model(nn.Module):

    def __init__(self, input_channel=4, output_dim=8):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channel, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.linear1 = nn.Linear(3136, 512)
        self.linear2 = nn.Linear(512, output_dim)

    def forward(self, x, mode='train'):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        batch = x.shape[0]
        x = x.view(batch, -1)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x, None, None

class MLP(nn.Module):

    def __init__(self, input_channel=args.predicate_num * (args.arity_num ** 2) , output_dim=7):
        super().__init__()

        self.l1 = nn.Linear(input_channel, 1024)
        self.l2 = nn.Linear(1024, 256)
        self.l3 = nn.Linear(256, output_dim)

    def forward(self, x, mode='train'):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        batch = x.shape[0]
        x = x.view(batch, -1)
        return x, None, None

class MetaPGPolicy(PGPolicy):
    
    def __init__(self,
        model,
        optim,
        mask=None,
        goal='meta',
        dist_fn=torch.distributions.categorical.Categorical,
        batch_size = 128,
        discount_factor=0.99,
        reward_normalization=False,
        device = 'cpu',
        repeat = 3,
        **kwargs):

        super().__init__(model, optim, dist_fn=torch.distributions.categorical.Categorical,
        discount_factor=0.99, reward_normalization=False,
        **kwargs)

        self.goal = goal

        self.device=device

        self.mask = mask

        self.buffer = ReplayBuffer(size=1000)

        self.success_ratio = -1

        self.batch_size = batch_size

        self.repeat = 3

    def forward(self, batch, state=None, **kwargs):

        obs = batch.obs
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs).float().to(self.device)
        
        logits, path_attn, pre_attn_list = self.model(obs)

        if self.mask != None:
            logits[:, self.mask] = torch.tensor(-float('inf')).to(self.device)
        
        logits = F.softmax(logits, -1)

        if isinstance(logits, tuple):
            dist = self.dist_fn(*logits)
        else:
            dist = self.dist_fn(logits)  # type: ignore
        act = dist.sample().detach().cpu().numpy()
        return Batch(logits=logits, act=act, state=None, dist=dist)

    def criticize(self, reachGoal, die, distanceReward, useSparseReward):
        reward = -0.1
        if reachGoal:
            reward += 10
            #reward += 50.0
        if die:
            reward -= 5
        if not useSparseReward:
            reward += distanceReward
        reward = np.minimum(reward, maxReward)
        reward = np.maximum(reward, minReward)
        return reward

    def add(self, obs, act, rew, done, obs_next):
        self.buffer.add(obs=obs, act=act, rew=rew, done=done, obs_next=obs_next)

    def update(self):
        batch_data, indice = self.buffer.sample(batch_size = 0)
        batch_data = self.process_fn(batch_data, self.buffer, indice)
        metric = self.learn(batch_data, 64, 3)
        self.buffer.reset()
        #self.buffer.anneal_beta(beta_schedule.value(step_count))
        return metric['loss']
    
    def save_weights(self, success_ratio):
        path = args.logdir + 'policy_subgoal_' + self.goal + '.pth'
        if success_ratio >= self.success_ratio:
            save_dict = {}
            
            if self.goal == 'meta':
                save_dict['meta'] = self.model.state_dict()
                save_dict['meta_optim'] = self.optim.state_dict()
            else:   
                save_dict['goal' + str(self.goal)] = self.model.state_dict()
                save_dict['goal' + str(self.goal) + '_optim'] = self.optim.state_dict()
                save_dict['eps' + str(self.goal)] = self.eps
            torch.save(save_dict, path)
            self.success_ratio = success_ratio

    def load_weights(self, logdir ,subgoal, best=False):

        if best:
            path = logdir + 'policy_subgoal_' + self.goal + '.pth'
            weights = torch.load(path, map_location=torch.device(self.device))
            if subgoal != 'meta':
                self.model.load_state_dict(weights['goal'+str(subgoal)])
                self.optim.load_state_dict(weights['goal'+str(subgoal) + '_optim'])
            else:
                self.model.load_state_dict(weights['meta'])
                self.optim.load_state_dict(weights['meta_optim'])

            
            print('Load weight successfully for goal ' + subgoal)
        else:
            import glob
            filelist = glob.glob(logdir + '*')
            for file_name in filelist:
                if 'last_checkpoint' in file_name:
                    weights = torch.load(file_name, map_location=torch.device(self.device))
                    if self.goal == 'meta':
                        self.model.load_state_dict(weights['meta'])
                        self.optim.load_state_dict(weights['meta_optim'])
                        print('Load weight successfully for meta controller ')
                        break
                    else:
                        self.model.load_state_dict(weights['goal' + subgoal])
                        self.optim.load_state_dict(weights['goal{0}_optim'.format(subgoal)])
                        self.set_eps(weights['eps' + subgoal])
                        print('Load weight successfully for goal {0} '.format(subgoal))
                        break
        if self._target:
            self.sync_weight()
 
    def lr_schedule(self):
        for p in self.optim.param_groups:
            p['lr'] *= 0.99

    def operate_buffer(self, option_learned):
        
        buffer = getattr(self, 'buffer', None)
        if option_learned:
            if buffer is not None:
                del self.buffer
        else:
            self.buffer = ReplayBuffer(size=1000)


def get_hdqn_obs(env):
    return env.getStackedState()

def get_nsrl_obs(env):
    return env.get_symbolic_tensor()

def get_mlp_obs(env):
    return env.get_symbolic_tensor().reshape((-1))

obs_fns = {'hdqn': get_hdqn_obs, 'mlp': get_mlp_obs, 'nsrl': get_nsrl_obs}

obs_fn = obs_fns[args.model]

model_fns = {'hdqn': lambda : Model(output_dim=7), 
            'mlp': lambda : MLP(output_dim=7), 
            'nsrl': lambda : MetaModel(predicate_num=args.predicate_num,
                                        arity_num=args.arity_num,
                                        steps=args.path_length,
                                        output_embedding_size=args.embedding_size,
                                        n_layers=args.n_layers,
                                        n_head=args.n_head)}

model_fn = model_fns[args.model]


def Sampleworker(parent, p, env_fn_wrapper):
    parent.close()
    env = env_fn_wrapper.data()
    try:
        while True:
            cmd, data = p.recv()
            if cmd == 'run_episode':
                p.send(env.run_episode())
            elif cmd == 'test':
                p.send(env.test())
            elif cmd == 'set_agent_eps':
                p.send(env.set_agent_eps(data))
            elif cmd == 'set_meta_eps':
                p.send(env.set_meta_eps(data))
            elif cmd == 'set_agent_weights':
                p.send(env.set_agent_weights(data))
            elif cmd == 'set_meta_weights':
                p.send(env.set_meta_weights(data))
            elif cmd == 'get_agent_eps':
                p.send(env.get_agent_eps())
            elif cmd == 'get_meta_eps':
                p.send(env.get_meta_eps())
            elif cmd == 'get_agent_weights':
                p.send(env.get_agent_weights())
            elif cmd == 'get_meta_weights':
                p.send(env.get_meta_weights())
            elif cmd =='get_rank':
                p.send(env.get_rank())
            elif cmd == 'get_meta_experience':
                p.send(env.get_meta_experience())
            elif cmd == 'get_agent_experience':
                p.send(env.get_agent_experience(data))
            elif cmd =='get_ratio_rew':
                p.send(env.get_ratio_rew())
            elif cmd == 'get_episode_step':
                p.send(env.get_episode_step())
            elif cmd == 'get_option_chosen_times':
                p.send(env.get_option_chosen_times())
            elif cmd == 'set_option_learned':
                p.send(env.set_option_learned(data))
            else:
                p.close()
                raise NotImplementedError
    except KeyboardInterrupt:
        p.close()

class SampleEnv():

    def __init__(self, args, rank ,env, agent_list, metaAgent, device='cpu', life=1):
        self.args = args
        self.agent_list = []

        self.agent_list = agent_list
        self.metaAgent = metaAgent
        self.env = env

        self.rank = rank
        self.life = life
    
    def reset(self):

        self.meta_experience = []
        self.agent_experience = {}
        self.episodeGoal_experience = {}
        self.option_chosen_times = [0 for _ in range(Num_subgoal)]
        for idx in range(Num_subgoal):
            self.agent_experience[idx] = []
            self.episodeGoal_experience[idx] = []
        
        self.steps = 0

        self.subgoal_success_tracker = [deque(maxlen=100) for _ in range(Num_subgoal)]

    def run_episode(self):

        for agent in self.agent_list:
            agent.train()
        
        self.metaAgent.train()

        self.reset()

        self.env.restart()

        args = self.args

        self.total_rew = 0

        episodeSteps = 0

        episodeGoalCount = 0

        while not self.env.isTerminal() and episodeSteps <= maxStepsPerEpisode and episodeGoalCount <= 7:
            #get corresbonding observations
            meta_obs = obs_fn(self.env)

            meta_batch = Batch(obs=[meta_obs])
    
            with torch.no_grad():

                goal = self.metaAgent(meta_batch).act[0]
            
            reach_goal = self.env.reach_goal()

            episodeGoal_experience = []

            agent_experience = []

            sub_externalRewards = 0

            subgoal = self.env.select_goal(goal, reach_goal)

            episodeGoalCount += 1

            if subgoal != -1:

                self.option_chosen_times[subgoal] += 1

                while not self.env.isTerminal() and not reach_goal == goal and episodeSteps <= maxStepsPerEpisode:

                    episodeSteps += 1

                    self.steps += 1

                    obs = self.env.getStackedState()

                    batch_data = Batch(obs=obs.reshape((1, 4, 84, 84)))
                
                    with torch.no_grad():
                        
                        result = self.agent_list[subgoal](batch_data)
                    
                    act = result.act[0]

                    tmp_rew = self.env.act(actionMap[act])

                    sub_externalRewards += tmp_rew

                    self.total_rew += tmp_rew

                    obs_next = self.env.getStackedState()

                    reach_goal = self.env.reach_goal()

                    done = self.env.isTerminal()

                    if tmp_rew == 100:
                        
                        reach_goal = 2
                    
                    
                    intrinsicRewards = self.agent_list[subgoal].criticize(reach_goal == goal, done, 0, True)

                    self.episodeGoal_experience[subgoal].append([obs, act, intrinsicRewards, done, obs_next])

                if episodeSteps > maxStepsPerEpisode or self.env.isTerminal():
                    self.subgoal_success_tracker[subgoal].append(0)

                elif reach_goal == goal:
                    self.subgoal_success_tracker[subgoal].append(1)

                    episodeSteps = 0

                    # The judgement of reaching goal is too strict for the agent to get huge reward
                    if goal == 1:
                        for _ in range(5):
                            tmp_rew += self.env.act(3)
                            self.total_rew += tmp_rew
                            sub_externalRewards += tmp_rew

                rew = sub_externalRewards
                act = goal
                done = self.env.isTerminal()
                rew = -10
                rew /= 20
                if args.model != 'hdqn' and (done or episodeSteps > maxStepsPerEpisode):
                    meta_obs_next = np.ones_like(meta_obs) * subgoal / 7
                    done = True
                else:
                    if reach_goal == goal and goal == 1:
                        meta_obs_next = np.zeros_like(meta_obs)
                        done = True
                    else:
                        meta_obs_next = obs_fn(self.env)

                self.meta_experience.append([meta_obs, act, rew, done, meta_obs_next])

                meta_obs = obs_fn(self.env)

            else:

                if args.model != 'hdqn':
                    meta_obs_next = np.ones_like(meta_obs) * subgoal / 7
                else:
                    meta_obs_next = obs_fn(self.env)

                self.meta_experience.append([meta_obs, goal, -150 / 20, True, meta_obs_next])
            
            if self.total_rew >= 400:
                break

        return None
        #return self.meta_experience, self.episodeGoal_experience, self.subgoal_success_tracker, self.total_rew

    def get_option_chosen_times(self):
        return self.option_chosen_times

    def get_meta_experience(self):
        return self.meta_experience
    
    def get_agent_experience(self, goal):
        return self.episodeGoal_experience[goal]

    def get_ratio_rew(self):
        return self.subgoal_success_tracker, self.total_rew

    def test(self):

        for agent in self.agent_list:
            agent.eval()
        
        self.metaAgent.eval()

        test_rew = 0

        self.env.restart()

        episodeSteps = 0

        while not self.env.isTerminal() and episodeSteps <= maxStepsPerEpisode:

            meta_obs = obs_fn(self.env)

            meta_batch = Batch(obs=[meta_obs])
        
            with torch.no_grad():

                goal = self.metaAgent(meta_batch, eps=0).act[0]

            reach_goal = self.env.reach_goal()

            subgoal = self.env.select_goal(goal, reach_goal)

            if subgoal != -1:
            
                while not self.env.isTerminal() and not reach_goal == goal and episodeSteps <= maxStepsPerEpisode:

                    obs = self.env.getStackedState()

                    episodeSteps += 1

                    batch_data = Batch(obs=obs.reshape((1, 4, 84, 84)), eps=0)

                    with torch.no_grad():
                        
                        result = self.agent_list[subgoal](batch_data)
                    
                    act = result.act[0]

                    tmp_rew = self.env.act(actionMap[act])

                    test_rew += tmp_rew

                    reach_goal = self.env.reach_goal()

                    if tmp_rew == 100:
                        reach_goal = 2
                
                if reach_goal == goal:
                    if goal == 1:
                        action = 3
                        for _ in range(5):
                            test_rew += self.env.act(action)
                
                if test_rew >= 400:
                    break
            else:
                break
            
        return test_rew

    def set_agent_eps(self, eps):
        for i,j in enumerate(eps):
            self.agent_list[i].set_eps(j)

    def set_meta_eps(self, eps):
        self.metaAgent.set_eps(eps)
    
    def set_agent_weights(self, state_dict):
        for i,j in enumerate(state_dict):
            self.agent_list[i].model.load_state_dict(j)

    def set_meta_weights(self, state_dict):
        self.metaAgent.model.load_state_dict(state_dict)

    def get_agent_eps(self):
        eps = [agent.eps for agent in self.agent_list]
        return eps
    
    def get_meta_eps(self):
        return self.metaAgent.eps
    
    def get_agent_weights(self):
        return [agent.model.state_dict() for agent in self.agent_list]
    
    def get_meta_weights(self):
        return self.metaAgent.model.state_dict()
    
    def get_rank(self):
        return self.rank

    def get_episode_step(self):
        return self.steps


    def set_option_learned(self,data):

        self.option_learned = data

class SubprocSampleEnv():
    """Vectorized environment wrapper based on subprocess.

    .. seealso::

        Please refer to :class:`~tianshou.env.BaseVectorEnv` for more detailed
        explanation.
    """
    def __init__(self, env_fns: List[Callable[[], gym.Env]]) -> None:

        self._env_fns = env_fns
        self.env_num = len(env_fns)

        self.closed = False
        self.parent_remote, self.child_remote = \
            zip(*[Pipe() for _ in range(self.env_num)])
        self.processes = [
            Process(target=Sampleworker, args=(
                parent, child, CloudpickleWrapper(env_fn)), daemon=True)
            for (parent, child, env_fn) in zip(
                self.parent_remote, self.child_remote, env_fns)
        ]
        for p in self.processes:
            p.start()
        for c in self.child_remote:
            c.close()

    def run_episode(self):
        
        id = range(self.env_num)
        for i in id:
            self.parent_remote[i].send(['run_episode', None])
        sample = [self.parent_remote[i].recv() for i in id]
        return sample
    
    def test(self):
        id = range(self.env_num)
        for i in id:
            self.parent_remote[i].send(['test', None])
        rew = [self.parent_remote[i].recv() for i in id]
        return rew

    def set_agent_eps(self, eps):
        
        id = range(self.env_num)
        for i in id:
            self.parent_remote[i].send(['set_agent_eps', eps])
        r = [self.parent_remote[i].recv() for i in id]
    
    def set_meta_eps(self, eps):
        id = range(self.env_num)
        for i in id:
            self.parent_remote[i].send(['set_meta_eps', eps])
        r = [self.parent_remote[i].recv() for i in id]

    def set_agent_weights(self, state_dict):
        id = range(self.env_num)
        for i in id:
            self.parent_remote[i].send(['set_agent_weights', state_dict])
        r = [self.parent_remote[i].recv() for i in id]
    
    def set_meta_weights(self, state_dict):
        id = range(self.env_num)
        for i in id:
            self.parent_remote[i].send(['set_meta_weights', state_dict])
        r = [self.parent_remote[i].recv() for i in id]

    def get_option_chosen_times(self):
        id = range(self.env_num)
        for i in id:
            self.parent_remote[i].send(['get_option_chosen_times', None])
        r = [self.parent_remote[i].recv() for i in id]
        return r        

    def get_agent_eps(self):
        
        id = range(self.env_num)
        for i in id:
            self.parent_remote[i].send(['get_agent_eps', None])
        r = [self.parent_remote[i].recv() for i in id]
        return r
  
    def get_meta_eps(self):
        
        id = range(self.env_num)
        for i in id:
            self.parent_remote[i].send(['get_meta_eps', None])
        r = [self.parent_remote[i].recv() for i in id]
        return r

    def get_agent_weights(self):
        
        id = range(self.env_num)
        for i in id:
            self.parent_remote[i].send(['get_agent_weights', None])
        r = [self.parent_remote[i].recv() for i in id]
        return r

    def get_meta_weights(self):
        
        id = range(self.env_num)
        for i in id:
            self.parent_remote[i].send(['get_meta_weights', None])
        r = [self.parent_remote[i].recv() for i in id]
        return r
    
    def get_rank(self):
        id = range(self.env_num)
        for i in id:
            self.parent_remote[i].send(['get_rank', None])
        r = [self.parent_remote[i].recv() for i in id]
        return r

    def get_meta_experience(self):
        id = range(self.env_num)
        for i in id:
            self.parent_remote[i].send(['get_meta_experience', None])
        r = [self.parent_remote[i].recv() for i in id]
        return r

    def get_agent_experience(self):
        id = range(self.env_num)
        
        data = []
        for i in id:
            data.append({})

        for i in id:
            for goal in range(7):
                self.parent_remote[i].send(['get_agent_experience', goal])
                data[i][goal] = self.parent_remote[i].recv()
        return data
    
    def get_ratio_rew(self):
        id = range(self.env_num)
        for i in id:
            self.parent_remote[i].send(['get_ratio_rew', i])
        r  = [self.parent_remote[i].recv() for i in id]
        return r
    
    def get_episode_step(self):
        id = range(self.env_num)
        for i in id:
            self.parent_remote[i].send(['get_episode_step', i])
        r  = [self.parent_remote[i].recv() for i in id]
        return r

    def set_option_learned(self, data):
        id = range(self.env_num)
        for i in id:
            self.parent_remote[i].send(['set_option_learned', data])
        r = [self.parent_remote[i].recv() for i in id]
        return r

'''
def TrainWorker(parent, p, model_wrapper):
    pass

class AsyncTrainer()
'''

def create_pair_model(agent_eps=None,
                      meta_eps=None,
                      agent_buffer_size=None,
                      agent_exploration_steps=None,
                      meta_buffer_size=0,
                      meta_exploration_steps=args.meta_explorationSteps,
                      device='cpu'):
    
    if agent_eps is None:
        agent_eps = [1.0 for _ in range(Num_subgoal)]
    
    if meta_eps is None:
        meta_eps = 1.0
    
    if agent_exploration_steps is None:
        agent_exploration_steps = [args.explorationSteps for _ in range(Num_subgoal)]
    
    if agent_buffer_size is None:
        agent_buffer_size = [0 for _ in range(Num_subgoal)]

    agent_list = []
    for indx in range(len(goal_to_train)):
        model = Model().to(device)
        opt = optim.RMSprop(model.parameters(), lr=args.lr, alpha=0.95, eps=1e-08)
        agent_list.append(DDQNPolicy(model=model,
                                    optim=opt, 
                                    goal=str(indx),
                                    train_freq=args.train_freq,
                                    discount_factor=args.gamma,
                                    estimation_step=10,
                                    target_update_freq=args.target_update_freq if agent_buffer_size[indx] else -1,
                                    reward_normalization=False,
                                    batch_size=args.batch,
                                    buffer_size=agent_buffer_size[indx],
                                    explorationSteps=agent_exploration_steps[indx],
                                    random_play_steps=args.random_steps,
                                    device=device,
                                    initial_eps=agent_eps[indx],
                                    final_eps=0.02
                                    ))
    
    metaModel = model_fn().to(device)
    metaOpt = optim.Adam(metaModel.parameters(), lr=args.lr)
    metaAgent = DDQNPolicy(model=metaModel, 
                        optim=metaOpt, 
                        goal='meta',
                        mask=[0],
                        train_freq=args.meta_train_freq,
                        discount_factor=args.gamma,
                        estimation_step=args.n_step,
                        target_update_freq=args.meta_target_update_freq if meta_buffer_size else -1,
                        reward_normalization=True,
                        batch_size=args.meta_batch,
                        random_play_steps=args.meta_random_steps,
                        buffer_size=meta_buffer_size,
                        explorationSteps=meta_exploration_steps,
                        device=device,
                        initial_eps=meta_eps,
                        final_eps=0.02)

    return agent_list, metaAgent

def create_pair_model_pg(agent_eps=None,
                      agent_buffer_size=None,
                      agent_exploration_steps=None,
                      meta_buffer_size=0,
                      meta_exploration_steps=args.meta_explorationSteps,
                      device='cpu'):
    
    if agent_eps is None:
        agent_eps = [1.0 for _ in range(Num_subgoal)]
    
    if agent_exploration_steps is None:
        agent_exploration_steps = [args.explorationSteps for _ in range(Num_subgoal)]
    
    if agent_buffer_size is None:
        agent_buffer_size = [0 for _ in range(Num_subgoal)]

    agent_list = []
    for indx in range(len(goal_to_train)):
        model = Model().to(device)
        opt = optim.RMSprop(model.parameters(), lr=args.lr, alpha=0.95, eps=1e-08)
        agent_list.append(DDQNPolicy(model=model,
                                    optim=opt, 
                                    goal=str(indx),
                                    train_freq=args.train_freq,
                                    discount_factor=args.gamma,
                                    estimation_step=5,
                                    target_update_freq=args.target_update_freq if agent_buffer_size[indx] else -1,
                                    reward_normalization=False,
                                    batch_size=args.batch,
                                    buffer_size=agent_buffer_size[indx],
                                    explorationSteps=agent_exploration_steps[indx],
                                    random_play_steps=args.random_steps,
                                    device=device,
                                    initial_eps=agent_eps[indx],
                                    final_eps=0.02
                                    ))
    
    metaModel = model_fn().to(device)
    metaOpt = optim.Adam(metaModel.parameters(), lr=args.lr)
    metaAgent = MetaPGPolicy(model=metaModel, 
                        optim=metaOpt, 
                        goal='meta',
                        mask=[0],
                        discount_factor=args.gamma,
                        reward_normalization=True,
                        batch_size=args.meta_batch,
                        buffer_size=meta_buffer_size,
                        device=device)

    return agent_list, metaAgent


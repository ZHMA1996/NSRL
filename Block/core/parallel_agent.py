import time
import os
import shutil
import numpy as np
import tianshou as ts

import copy
import torch
import torch.nn as nn

from tianshou.data import Batch
from tianshou.data import to_torch_as
from tianshou.data import to_numpy
from tianshou.policy import DQNPolicy, PGPolicy, PPOPolicy
from tianshou.data import PrioritizedReplayBuffer, ReplayBuffer

import torch.optim as optim
from torch.multiprocessing import Process, Pipe
from typing import List, Tuple, Union, Optional, Callable, Any

from tianshou.utils import tqdm_config


from core.modules import *
from core.argparser import *
from core.symbolicEnvironment import *

import gym
import random
import pickle as pk
import torch.distributions as dist

lr = args.lr

class MetaPPOPolicy(PPOPolicy):
    
    def __init__(self,
        actor,
        critic,
        optim,
        dist_fn=torch.distributions.categorical.Categorical,
        discount_factor=0.99,
        max_grad_norm=None,
        eps_clip=0.2,
        vf_coef=0.5,
        ent_coef=0.01,
        action_range=None,
        gae_lambda=0.95,
        dual_clip=None,
        value_clip=True,
        reward_normalization=False,
        max_batchsize=256,
        device = 'cpu',
        **kwargs):

        super().__init__(actor, critic, optim, dist_fn=torch.distributions.categorical.Categorical,
        discount_factor=0.99, max_grad_norm=None, eps_clip=0.2, vf_coef=0.5, ent_coef=0.01,
        action_range=None, gae_lambda=0.95, dual_clip=None, value_clip=True,
        reward_normalization=False, max_batchsize=256,
        **kwargs)

        self.device=device

    def forward(self, batch, state=None, mode='train', **kwargs):

        obs = batch.obs
        mask = obs[:,-1,:]
        obs = obs[:,:-1,:]
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs).float().to(self.device)

        if mode == 'train':
            logits = self.actor(obs)
        else:
            logits, path_attn, pre_attn_list = self.actor(obs, mode='test')
            path_attn = path_attn.cpu().numpy()
            for i, pre_attn in enumerate(pre_attn_list):
                pre_attn_list[i] = pre_attn.cpu().numpy()

        for i in range(len(mask)):
            ind = np.where(mask[i] == 0)
            logits[i,ind] = torch.tensor(-float('inf')).to(self.device)
        logits = F.softmax(logits, dim=-1)
        if isinstance(logits, tuple):
            dist = self.dist_fn(*logits)
        else:
            dist = self.dist_fn(logits)  # type: ignore
        act = dist.sample().detach().cpu().numpy()
        if mode == 'train':
                return Batch(logits=logits, act=act, state=None, dist=dist)
        else:
            return Batch(logits=logits, act=act, state=None, dist=dist, path_attn=path_attn, pre_attn_list=pre_attn_list)

    def save_weights(self, prefix=None):

        state_dict = {}
        state_dict['actor'] = self.actor.state_dict()
        state_dict['critic'] = self.critic.state_dict()
        state_dict['optim'] = self.optim.state_dict()
        path = args.logdir + args.task
        path += ('_' + prefix + '.pth') if prefix else ('.pth')
        torch.save(state_dict, path)

    def load_weights(self, logdir, prefix=None):
        path = logdir + args.task
        path += ('_' + prefix + '.pth') if prefix else ('.pth')
        weights = torch.load(path, map_location=torch.device(self.device))
        self.actor.load_state_dict(weights['actor'])
        self.critic.load_state_dict(weights['critic'])
        self.optim.load_state_dict(weights['optim'])
        print('Load weight successfully')

    def lr_schedule(self):
        for p in self.optim.param_groups:
            p['lr'] *= 0.99

    def learn(self, batch, batch_size, repeat):
        self._batch = batch_size
        losses, clip_losses, vf_losses, ent_losses = [], [], [], []
        v = []
        old_log_prob = []
        with torch.no_grad():
            for b in batch.split(batch_size, shuffle=False):
                v.append(self.critic(b.obs))
                old_log_prob.append(self(b).dist.log_prob(
                    to_torch_as(b.act, v[0])))
        batch.v = torch.cat(v, dim=0).squeeze(-1)  # old value
        batch.act = to_torch_as(batch.act, v[0])
        batch.logp_old = torch.cat(old_log_prob, dim=0).reshape(batch.v.shape)
        batch.returns = to_torch_as(batch.returns, v[0])
        if self._rew_norm:
            mean, std = batch.returns.mean(), batch.returns.std()
            if not np.isclose(std.item(), 0):
                batch.returns = (batch.returns - mean) / std
        batch.adv = batch.returns - batch.v
        if self._rew_norm:
            mean, std = batch.adv.mean(), batch.adv.std()
            if not np.isclose(std.item(), 0):
                batch.adv = (batch.adv - mean) / std
        for _ in range(repeat):
            for b in batch.split(batch_size):
                dist = self(b).dist
                value = self.critic(b.obs).squeeze(-1)
                ratio = (dist.log_prob(b.act).reshape(value.shape) - b.logp_old
                         ).exp().float()
                surr1 = ratio * b.adv
                surr2 = ratio.clamp(
                    1. - self._eps_clip, 1. + self._eps_clip) * b.adv
                if self._dual_clip:
                    clip_loss = -torch.max(torch.min(surr1, surr2),
                                           self._dual_clip * b.adv).mean()
                else:
                    clip_loss = -torch.min(surr1, surr2).mean()
                clip_losses.append(clip_loss.item())
                if self._value_clip:
                    v_clip = b.v + (value - b.v).clamp(
                        -self._eps_clip, self._eps_clip)
                    vf1 = (b.returns - value).pow(2)
                    vf2 = (b.returns - v_clip).pow(2)
                    vf_loss = .5 * torch.max(vf1, vf2).mean()
                else:
                    vf_loss = .5 * (b.returns - value).pow(2).mean()
                vf_losses.append(vf_loss.item())
                e_loss = dist.entropy().mean()
                ent_losses.append(e_loss.item())
                loss = clip_loss + self._w_vf * vf_loss
                #loss = clip_loss + self._w_vf * vf_loss - self._w_ent * e_loss
                losses.append(loss.item())
                self.optim.zero_grad()
                loss.backward()
                '''
                nn.utils.clip_grad_norm_(list(
                    self.actor.parameters()) + list(self.critic.parameters()),
                    self._max_grad_norm)
                '''
                self.optim.step()
        return {
            'loss': losses,
            'loss/clip': clip_losses,
            'loss/vf': vf_losses,
            'loss/ent': ent_losses,
        }

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

        self.linear = nn.Linear(arity_num ** 2, arity_num ** 2)


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

        score = self.linear(final_path).reshape((batch_size,-1))



        if mode == 'train':
            return score
        else:
            return score, path_attn, pre_attn_list

def create_nsrl(block_n=4,
                device='cpu',
                model_type='chain'):
    
    metaModel = MetaModel(  predicate_num=args.predicate_num,
                                arity_num=args.arity_num, 
                                steps=args.path_length, 
                                output_embedding_size=args.embedding_size,
                                n_layers=args.n_layers,
                                n_head=args.n_head).to(device)


    metaOpt = optim.Adam(metaModel.parameters(), lr=args.lr)
    metaAgent = MetaPGpolicy(model=metaModel, 
                        optim=metaOpt,
                        discount_factor=args.gamma,
                        reward_normalization=False,
                        device=device)

    return metaAgent

class ValueModel(nn.Module):

    def __init__(self, device='cpu'):
        super().__init__()
        self.l1 = nn.Linear(args.predicate_num *(args.arity_num ** 2), 20)
        self.l2 = nn.Linear(20, 1)
        self.device = device
 
    def forward(self, x):
        x = x[:,:-1,:]
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x).float().to(self.device)
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x


def create_nsrl_ppo(device='cpu', model_type='chain'):
    critic = ValueModel(device).to(device)
    actor = MetaModel(  predicate_num=args.predicate_num,
                            arity_num=args.arity_num, 
                            steps=args.path_length, 
                            output_embedding_size=args.embedding_size,
                            n_layers=args.n_layers,
                            n_head=args.n_head).to(device)

    opt = optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=args.lr)
    metaAgent = MetaPPOPolicy(actor=actor,
                        critic=critic,
                        optim=opt,
                        discount_factor=args.gamma,
                        reward_normalization=False,
                        device=device)
    return metaAgent

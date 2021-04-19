from __future__ import print_function, division, absolute_import
from core.clause import *
import copy
import random
from random import choice
from core.argparser import *

from itertools import permutations

from tianshou.data import Batch

import time

class SymbolicEnvironment(object):
    def __init__(self, background, initial_state, actions):
        '''
        :param language_frame
        :param background: list of atoms, the background knowledge
        :param positive: list of atoms, positive instances
        :param negative: list of atoms, negative instances
        '''
        self.background = background
        self._state = copy.deepcopy(initial_state)
        self.initial_state = copy.deepcopy(initial_state)
        self.actions = actions
        self.acc_reward = 0
        self.episodeStep = 0

    def reset(self):
        self.acc_reward = 0
        self.episodeStep = 0
        self._state = copy.deepcopy(self.initial_state)

ON = Predicate("on", 2)
TOP = Predicate("top", 1)
MOVE = Predicate("move", 2)
INI_STATE = [["a", "b", "c", "d"]]
INI_STATE2 = [["a"], ["b"], ["c"], ["d"]]
FLOOR = Predicate("floor", 1)
BLOCK = Predicate("block", 1)
CLEAR = Predicate("clear", 1)
MAX_WIDTH = 7

import numpy as np
block_encoding = {"a":1, "b": 2, "c":3, "d":4, "e": 5, "f":6, "g":7}
idx2block = {}
for key, value in block_encoding.items():
    idx2block[value] = key


def sample(num_list, number):

    res = []

    for _ in range(number):

        ind = random.choice(range(len(num_list)))
        res.append(num_list[ind])
        num_list.pop(ind)
    
    return res

import string

class BlockWorld(SymbolicEnvironment):
    """
    state is represented as a list of lists
    """
    def __init__(self, initial_state=INI_STATE, additional_predicates=(), background=(), block_n=4, all_block=False, test=False):
        actions = [MOVE]
        self.max_step = 50
        self._block_encoding = {"a":1, "b": 2, "c":3, "d":4, "e": 5, "f":6, "g":7}

        self.idx2block = {}
        for key, value in self._block_encoding.items():
            self.idx2block[value] = key

        self.state_dim = MAX_WIDTH**3
        if all_block:
            self._all_blocks = list(string.ascii_lowercase)[:MAX_WIDTH]+["floor"]
        else:
            self._all_blocks = list(string.ascii_lowercase)[:block_n]+["floor"]

        self._additional_predicates = additional_predicates
        background = list(background)
        background.append(Atom(FLOOR, ["floor"]))
        #background.extend([Atom(BLOCK, [b]) for b in list(string.ascii_lowercase)[:block_n]])
        super(BlockWorld, self).__init__(background, initial_state, actions)
        self._block_n = block_n

        self.test = test

        self.mask = None

        self.test_state = initial_state

        self.random_initial_state()

    def clean_empty_stack(self):
        self._state = [stack for stack in self._state if stack]

    @property
    def all_actions(self):
        return [Atom(MOVE, [a, b]) for a in self._all_blocks for b in self._all_blocks]

    @property
    def state(self):
        return tuple([tuple(stack) for stack in self._state])

    def step(self, action):
        """
        :param action: action is a ground atom
        :return:
        """

        action = self.compute_atoms(action)

        self.episodeStep +=1
        self.clean_empty_stack()
        block1, block2 = action.terms
        
        reward, finished = self.get_reward()
        self.acc_reward += reward
        if finished and reward<1:
            self._state = [[]]
            s = self.get_symbolic_tensor(self._state)
            return np.vstack((self.get_symbolic_tensor(self._state), self.mask)), reward, finished, {}
        for stack1 in self._state:
            if stack1[-1] == block1:
                for stack2 in self._state:
                    if stack2[-1] == block2:
                        del stack1[-1]
                        stack2.append(block1)
                        s = self.get_symbolic_tensor(self._state)
                        return np.vstack((self.get_symbolic_tensor(self._state), self.mask)), reward, finished, {}
        if block2 == "floor":
            for stack1 in self._state:
                if stack1[-1] == block1 and len(stack1)>1:
                    del stack1[-1]
                    self._state.append([block1])
                    s = self.get_symbolic_tensor(self._state)
                    return np.vstack((self.get_symbolic_tensor(self._state), self.mask)), reward, finished, {}
        #s = self.get_symbolic_tensor(self._state)
        return np.vstack((self.get_symbolic_tensor(self._state), self.mask)), reward, finished, {}

    @property
    def action_n(self):
        return (self._block_n+1)**2

    def get_symbolic_tensor(self, state):
        
        matrix = np.zeros(([2, MAX_WIDTH + 1, MAX_WIDTH + 1]))

        '''
        Predicate Design
        0 : On
        1 : Top

        Object Design
        0 : Floor
        1 : Block 1
        2 : Block 2
        3 : Block 3
        4 : Block 4
        5 : Block 5
        6 : Block 6
        7 : Block 7
        '''
        for i, stack in enumerate(state):
            for j in range(len(stack)):
                block = stack[j]
                if j == len(stack) - 1:
                    matrix[1][self._block_encoding[block]][self._block_encoding[block]] = 1
                if j == 0:
                    matrix[0][self._block_encoding[block]][0] = 1
                else: 
                    matrix[0][self._block_encoding[block]][self._block_encoding[stack[j-1]]] = 1
        
        return matrix.reshape((2, 8 * 8))

    def state2vector(self, state):
        matrix = np.zeros([MAX_WIDTH, MAX_WIDTH, MAX_WIDTH])
        for i, stack in enumerate(state):
            for j, block in enumerate(stack):
                matrix[i][j][self._block_encoding[block]-1] = 1.0
        return matrix.flatten()

    def state2atoms(self, state):
        atoms = set()
        for stack in state:
            if len(stack)>0:
                atoms.add(Atom(ON, [stack[0], "floor"]))
                atoms.add(Atom(TOP, [stack[-1]]))
            for i in range(len(stack)-1):
                atoms.add(Atom(ON, [stack[i+1], stack[i]]))
        return atoms

    def compute_atoms(self, act):
        
        from math import floor
        row = floor(act / 8)
        col = act - row * 8

        if row:
            row = self.idx2block[row]
        else:
            row = 'floor'

        if col:
            col = self.idx2block[col]
        else:
            col = 'floor' 

        action_atom = Atom(MOVE, [row,col])

        return action_atom

    def reset(self):
        self.acc_reward = 0
        self.episodeStep = 0
        self._state = copy.deepcopy(self.initial_state)

    def close(self):
        pass

    def compute_mask(self, block_list):
        
        mask = [0 for _ in range(64)]

        block_list.append(0)
        block_list = sorted(block_list)

        for i in block_list:
            for j in block_list:
                mask[i*8+j] = 1
        
        self.mask = np.array(mask).reshape((1,-1))

    def set_test(self, test):

        self.test = test

class Unstack(BlockWorld):
    all_variations = ("swap top 2","2 columns", "5 blocks",
                      "6 blocks", "7 blocks")

    nn_variations = ("swap top 2","2 columns", "5 blocks", "6 blocks", "7 blocks")
    def get_reward(self):
        if self.episodeStep >= self.max_step:
            return -0.0, True
        for stack in self._state:
            if len(stack) > 1:
                return -0.02, False
        return 1.0, True

    def vary(self, type, test=False):
        block_n = self._block_n
        self.test = True
        if type=="swap top 2":
            initial_state=[["a", "b", "d", "c"]]
        elif type=="2 columns":
            initial_state=[["b", "a"], ["c", "d"]]
        elif type=="5 blocks":
            initial_state=[["a", "b", "c", "d", "e"]]
            block_n = 5
        elif type=="6 blocks":
            initial_state=[["a", "b", "c", "d", "e", "f"]]
            block_n = 6
        elif type=="7 blocks":
            initial_state=[["a", "b", "c", "d", "e", "f", "g"]]
            block_n = 7
        else:
            raise ValueError
        return Unstack(initial_state, self._additional_predicates, self.background, block_n, False, test)

    def reset(self):

        if self.test:
            tmp = np.array(self.test_state).reshape((-1)).tolist()
            block_list = [block_encoding[i] for i in tmp]
            self.initial_state = self.test_state
            super().reset()
        else:
            
            block_list = self.initial_state_set[self.state_index]

            self.state_index = (self.state_index + 1) % len(self.initial_state_set)

            '''
            block_list = [1,2,4,3]
            while block_list[0] == 1 and block_list[1] == 2 and block_list[2] == 4 and block_list[3] == 3:
                block_list = sample([i for i in range(1,8)], 4)
            '''
            
            state = [[idx2block[i] for i in block_list]]
            self.initial_state = state
            self.acc_reward = 0
            self.episodeStep = 0
            self._state = copy.deepcopy(state)

        self.compute_mask(block_list)

        return np.vstack((self.get_symbolic_tensor(self._state), self.mask))
        #return self.get_symbolic_tensor(self._state), block_list

    def random_initial_state(self):
        
        self.initial_state_set = []

        ind = [i for i in range(1,8)]

        for block_list in permutations(ind, 4):
            if block_list[0] == 1 and block_list[1] == 2 and block_list[2] == 4 and block_list[3] == 3:
                continue
            self.initial_state_set.append(copy.deepcopy(list(block_list)))
        
        self.state_index = 0

class Stack(BlockWorld):
    all_variations = ("swap right 2","2 columns", "5 blocks",
                      "6 blocks", "7 blocks")
    nn_variations = ("swap right 2","2 columns", "5 blocks", "6 blocks", "7 blocks")

    def get_reward(self):
        if self.episodeStep >= self.max_step:
            return -0.0, True
        for stack in self._state:
            if len(stack) == self._block_n:
                return 1.0, True
        return -0.02, False

    def vary(self, type, test=False):
        block_n = self._block_n
        self.test = True
        if type=="swap right 2":
            initial_state=[["a"], ["b"], ["d"], ["c"]]
        elif type=="2 columns":
            initial_state=[["b", "a"], ["c", "d"]]
        elif type=="5 blocks":
            initial_state=[["a"], ["b"], ["c"], ["d"], ["e"]]
            block_n = 5
        elif type=="6 blocks":
            initial_state=[["a"], ["b"], ["c"], ["d"], ["e"], ["f"]]
            block_n = 6
        elif type=="7 blocks":
            initial_state=[["a"], ["b"], ["c"], ["d"], ["e"], ["f"], ["g"]]
            block_n = 7
        else:
            raise ValueError
        return Stack(initial_state, self._additional_predicates, self.background, block_n, False, test)

    def reset(self):

        if self.test:
            tmp = np.array(self.test_state).reshape((-1)).tolist()
            block_list = [block_encoding[i] for i in tmp]
            self.initial_state = self.test_state
            super().reset()
        else:
            #block_list = np.random.choice(range(1,8), size=4, replace=False).tolist()
            
            '''
            block_list = [1,2,4,3]
            while block_list[0] == 1 and block_list[1] == 2 and block_list[2] == 4 and block_list[3] == 3:
                block_list = sample([i for i in range(1,8)], 4)
            '''

            block_list = self.initial_state_set[self.state_index]

            self.state_index = (self.state_index + 1) % len(self.initial_state_set)

            state = []
            for i in block_list:
                state.append([idx2block[i]])
            
            self.initial_state = state
            self.acc_reward = 0
            self.episodeStep = 0
            self._state = copy.deepcopy(state)
        
        self.compute_mask(block_list)

        return np.vstack((self.get_symbolic_tensor(self._state), self.mask))

    def random_initial_state(self):
        
        
        self.initial_state_set = []

        ind = [i for i in range(1,8)]

        for block_list in permutations(ind, 4):
            if block_list[0] == 1 and block_list[1] == 2 and block_list[2] == 4 and block_list[3] == 3:
                continue
            self.initial_state_set.append(copy.deepcopy(list(block_list)))
        
        self.state_index = 0


GOAL_ON = Predicate("goal_on", 2)
class On(BlockWorld):
    all_variations = ("swap top 2","swap middle 2", "5 blocks",
                      "6 blocks", "7 blocks")
    nn_variations = ("swap top 2","swap middle 2")
    def __init__(self, initial_state=INI_STATE, goal_state=Atom(GOAL_ON, ["a", "b"]), block_n=4, all_block=False, test=False):
        super(On, self).__init__(initial_state, additional_predicates=[],
                                 background=[goal_state], block_n=block_n, all_block=all_block, test=test)
        self.goal_state = goal_state

        self.random_initial_state()

    def get_reward(self):
        if self.episodeStep >= self.max_step:
            return 0.0, True
        if Atom(ON, self.goal_state.terms) in self.state2atoms(self._state):
            return 1.0, True
        return -0.02, False

    def vary(self, type, test=False):
        block_n = self._block_n
        self.test = True
        if type=="swap top 2":
            initial_state=[["a", "b", "d", "c"]]
        elif type=="swap middle 2":
            initial_state=[["a", "c", "b", "d"]]
        elif type=="5 blocks":
            initial_state=[["a", "b", "c", "d", "e"]]
            block_n = 5
        elif type=="6 blocks":
            initial_state=[["a", "b", "c", "d", "e", "f"]]
            block_n = 6
        elif type=="7 blocks":
            initial_state=[["a", "b", "c", "d", "e", "f", "g"]]
            block_n = 7
        else:
            raise ValueError
        return On(initial_state=initial_state, block_n=block_n, all_block=False,test=test)

    def get_symbolic_tensor(self, state):
        
        mat = super().get_symbolic_tensor(state).reshape((2,8,8))
        if args.goal_on:
            matrix = np.zeros(([3, MAX_WIDTH + 1, MAX_WIDTH + 1]))
            matrix[2][1][2] = 1
            matrix[:2,:,:] = mat
            return matrix.reshape((3,8*8))
        else:
            return mat.reshape((2,8*8))

    def reset(self):

        if self.test:
            tmp = np.array(self.test_state).reshape((-1)).tolist()
            block_list = [block_encoding[i] for i in tmp]
            self.initial_state = self.test_state
            super().reset()

        else:
            
            block_list = [0]

            while 0 in block_list:

                block_list = self.initial_state_set[self.state_index]

                self.state_index = (self.state_index + 1) % len(self.initial_state_set)

            state = [[idx2block[i] for i in block_list]]

            '''
            block_list = [1,3,2,4]

            while (block_list[0] == 1 and block_list[1] == 3 and block_list[2] == 2 and block_list[3] == 4) or \
                    (block_list[0] == 1 and block_list[1] == 2 and block_list[2] == 4 and block_list[3] == 3):
                block_list = [2]
                block_list.extend(sample([i for i in range(3,8)], 2))
                random.shuffle(block_list)
                block_list.insert(0,1)

            block_list = [1,3,2,4]
            
            a = 2
            b = 1

            while (block_list[0] == 1 and block_list[1] == 3 and block_list[2] == 2 and block_list[3] == 4) or \
                    (block_list[0] == 1 and block_list[1] == 2 and block_list[2] == 4 and block_list[3] == 3) or b == a - 1:
                block_list = [1,2]
                block_list.extend(sample([i for i in range(3,8)], 2))
                random.shuffle(block_list)

                a = block_list.index(1)
                b = block_list.index(2)
            '''

            self.initial_state = state
            self.acc_reward = 0
            self.episodeStep = 0
            self._state = copy.deepcopy(state)
        
        self.compute_mask(block_list)

        return np.vstack((self.get_symbolic_tensor(self._state), self.mask))


    def random_initial_state(self):
        
        
        self.initial_state_set = []

        ind = [3,4,5,6,7]

        block_list = [1,2,3,4]

        iter_list = [2,3,4]

        for i in range(len(ind)):
            iter_list[1] = ind[i]
            for j in range(i+1, len(ind)):
                iter_list[2] = ind[j]
                for blocks in permutations(iter_list, 3):
                    block_list = [1]
                    block_list.extend(blocks)
                    if (block_list[0] == 1 and block_list[1] == 3 and block_list[2] == 2 and block_list[3] == 4) or \
                        (block_list[0] == 1 and block_list[1] == 2 and block_list[2] == 4 and block_list[3] == 3):
                        continue
                    self.initial_state_set.append(copy.deepcopy(block_list))
        
        self.state_index = 0

        '''
        block_list = [1,3,2,4]

        while (block_list[0] == 1 and block_list[1] == 3 and block_list[2] == 2 and block_list[3] == 4) or \
                (block_list[0] == 1 and block_list[1] == 2 and block_list[2] == 4 and block_list[3] == 3):
            block_list = [2]
            block_list.extend(sample([i for i in range(3,8)], 2))
            random.shuffle(block_list)
            block_list.insert(0,1)
        
        if block_list not in self.initial_state_set:
            self.initial_state_set.append(block_list)
        '''
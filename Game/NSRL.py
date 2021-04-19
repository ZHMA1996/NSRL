from argParser import *
from agent import *
from env import *

import torch.optim as optim

import torch
import tqdm
import time
import pickle
import numpy as np
from runx.logx import logx

from collections import Counter

seed = 1337
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


actionMap = [0, 1, 2, 3, 4, 5, 11, 12]
goal_to_train = [i for i in range(7)]
Num_subgoal = len(goal_to_train)
maxStepsPerEpisode = 500
logx.initialize(logdir=args.logdir, hparams=vars(args), tensorboard=True)


env_list = []
num = args.num_process
rank_list = [i for i in range(num)]
for i in range(num):
    p, q = create_pair_model(device='cpu')
    env_list.extend([lambda : SampleEnv(args, rank_list[i], ALEEnvironment(args.game, args) ,p, q)])

env = SubprocSampleEnv(env_list)
agent_eps = [1.0 for _ in range(Num_subgoal)]
agent_exploration_steps = [args.explorationSteps for _ in range(Num_subgoal)]


agent_exploration_steps[0] = 60000
agent_exploration_steps[1] = 120000
agent_exploration_steps[2] = 60000
agent_exploration_steps[3] = 60000
agent_exploration_steps[4] = 120000
agent_exploration_steps[5] = 60000
agent_exploration_steps[6] = 60000

save_learned = [False for _ in range(8)]

agent_buffer_size = [args.buffer_size for _ in range(Num_subgoal)]

agent_list, metaAgent = create_pair_model(  agent_eps=agent_eps,
                                            meta_eps=1.0,
                                            agent_buffer_size=agent_buffer_size,
                                            agent_exploration_steps=agent_exploration_steps,
                                            meta_buffer_size=args.meta_buffer_size,
                                            meta_exploration_steps=args.meta_explorationSteps,
                                            device='cuda:'+str(args.gpu) if args.gpu >=0 and torch.cuda.is_available() else 'cpu')


option_meta = 0
option_t = [0 for _ in range(Num_subgoal)]
success_tracker = [[] for _ in range(Num_subgoal)]

sample_meta = 0
sample_t = [0 for _ in range(Num_subgoal)]
option_performance = [0 for _ in range(Num_subgoal)]

option_learned = [False for _ in range(Num_subgoal)]


#option_learned = [True for _ in range(Num_subgoal)]
#option_learned[-1] = False


option_start_train_episode = [None for _ in range(Num_subgoal)]
meta_start_train_episode = None

Steps = 0
test_rew = 0
rew_num = 0
episodeCount = 0
cumulative_average_reward = 0
test_external_rew = 0

data = {}
data['trainGoal/sample_used'] = []
data['trainGoal/success_ratio'] = []
data['trainGoal/loss'] = []
data['trainGoal/start_train'] = []
data['trainGoal/chosen_times'] = []

data['trainMeta/sample_used'] = []
data['trainMeta/external_rew'] = []
data['trainMeta/average_rew'] = []
data['trainMeta/subenv_external_rew'] = []
data['trainMeta/loss'] = []
data['trainMeta/start_train'] = None

data['testMeta/external_rew'] = []
data['testMeta/average_rew'] = []

data['trainMeta/steps'] = []
data['testMeta/steps'] = []

def sync_process_weights():
    # transfer weights to sub process
    agents_state_dict = [agent.model.state_dict() for agent in agent_list]
    meta_state_dict = metaAgent.model.state_dict()

    for agent_state_dict in agents_state_dict:
        for key, value in agent_state_dict.items():
            agent_state_dict[key] = value.detach().to('cpu')
    for key, value in meta_state_dict.items():
        meta_state_dict[key] = value.detach().to('cpu')

    # set weights
    env.set_agent_weights(agents_state_dict)
    env.set_meta_weights(meta_state_dict)

    # set eps
    env.set_agent_eps([agent.eps for agent in agent_list])

    env.set_meta_eps(metaAgent.eps)
    
    # set option_learned
    env.set_option_learned(option_learned)

# sync weights initially
sync_process_weights()

if args.mode == 'train':

    with tqdm.tqdm(total=args.episode_limit, desc='Train', **tqdm_config) as t:

        while episodeCount <= t.total:

            env.set_option_learned(option_learned)

            sample = env.run_episode()

            meta_data = env.get_meta_experience()

            agent_data = env.get_agent_experience()

            ratio_rew = env.get_ratio_rew()

            episode_step = env.get_episode_step()

            Steps += sum(episode_step)

            metric_rew = []

            external_rew = 0

            option_chosen_times = env.get_option_chosen_times()

            for goal in range(7):
                agent_list[goal].operate_buffer(option_learned[goal])

            for idx in range(num):#every env

                meta_experience = meta_data[idx]

                agent_experience = agent_data[idx]

                subgoal_success_tracker, reward = ratio_rew[idx]

                #meta_experience, agent_experience, subgoal_success_tracker, reward = sample[idx]

                external_rew += reward

                metric_rew.append(reward)

                for experience in meta_experience:

                    obs, act, rew, done, obs_next = experience

                    metaAgent.add(obs=obs, act=act, rew=rew, done=done, obs_next=obs_next)

                    option_meta += 1

                for key in agent_experience.keys():
                    
                    option_t[key] += len(agent_experience[key])

                    if option_learned[key]:
                        continue

                    for experience in agent_experience[key]:

                        obs, act, rew, done, obs_next = experience

                        agent_list[key].add(obs=obs, act=act, rew=rew, done=done, obs_next=obs_next)

                for i in range(len(subgoal_success_tracker)):

                    success_tracker[i].extend(subgoal_success_tracker[i])
            
            logx.msg('Reward: ' + str(dict(Counter(metric_rew))))

            episodeCount += num

            # compute train metrics
            data['trainMeta/steps'].append(Steps)
            data['trainMeta/subenv_external_rew'].append(metric_rew)
            externalRewards = external_rew / num
            data['trainMeta/external_rew'].append(externalRewards)
            rew_num += 1
            cumulative_average_reward = (cumulative_average_reward * (rew_num - 1) + externalRewards) / rew_num

            # save train metrics
            logx.add_scalar('trainMeta/train_rew/episode', externalRewards, episodeCount)
            logx.add_scalar('trainMeta/train_rew/steps', externalRewards, Steps)
            logx.add_scalar('trainMeta/cumulative_average_rew/episode', cumulative_average_reward, episodeCount)
            logx.add_scalar('trainMeta/cumulative_average_rew/steps', cumulative_average_reward, Steps)

            # save train weights
            save_dict = {}
            save_dict['meta'] = metaAgent.model.state_dict()
            save_dict['meta_optim'] = metaAgent.optim.state_dict()
            save_dict['meta_eps'] = metaAgent.eps
            
            for goal in goal_to_train:
                save_dict['goal' + str(goal)] = agent_list[goal].model.state_dict()
                save_dict['eps' + str(goal)] = agent_list[goal].eps
                save_dict['goal' + str(goal) + '_optim'] = agent_list[goal].optim.state_dict()

            logx.save_model(save_dict, cumulative_average_reward, episodeCount)

            # save best subgoal weights based on success ratio
            for goal in goal_to_train:
                if len(success_tracker[goal]) >= 100:
                    option_performance[goal] = sum(success_tracker[goal][-100:]) / 100.
                    if option_performance[goal] >= args.stop_threshold:
                        option_learned[goal] = True

                        agent_list[goal].save_weights(option_performance[goal])


                    else:
                        option_learned[goal] = False
                else:
                    option_performance[goal] = 0.

                # save best weights

                # save metrics
                logx.add_scalar('trainGoal/' + idx2loc[goal] + '/success_ratio', option_performance[goal], Steps)
                logx.add_scalar('trainGoal/' + idx2loc[goal] + '/success_ratio', option_performance[goal], Steps)


            num_option_learned = sum(option_learned)
            if num_option_learned and  not save_learned[num_option_learned-1]:
                torch.save(save_dict, args.logdir + 'save_dict_{0}.pth'.format(str(num_option_learned)))
                save_learned[num_option_learned-1] = True


            meta_loss = []
            option_loss = [[] for _ in range(Num_subgoal)]

            # train subgoal network
            for goal in goal_to_train:
                agent_list[goal].train()
                if option_t[goal] >= agent_list[goal].random_play_steps and not option_learned[goal] and len(agent_list[goal].buffer) > agent_list[goal].batch_size:
                    for _ in range(args.controller_train_times):
                        sample_t[goal] += agent_list[goal].batch_size
                        loss = agent_list[goal].update()
                        option_loss[goal].append(loss)
                    if option_start_train_episode[goal] == None:
                        option_start_train_episode[goal] = episodeCount
                    
                    logx.add_scalar('trainGoal/' + str(goal) + '/loss', np.mean(option_loss[goal]), episodeCount)
                    logx.add_scalar('trainGoal/' + str(goal) + '/loss', np.mean(option_loss[goal]), Steps)

            # train meta network
            if option_meta > metaAgent.random_play_steps:
                metaAgent.train()
                if meta_start_train_episode == None:
                    meta_start_train_episode = episodeCount
                for _ in range(args.meta_train_times):
                    loss = metaAgent.update()
                    sample_meta += args.meta_batch
                    meta_loss.append(loss)

                logx.add_scalar('trainMeta/loss/episode', np.mean(meta_loss), episodeCount)
                logx.add_scalar('trainMeta/loss/steps', np.mean(meta_loss), Steps)

            # anneal subgoal eps
            for goal in goal_to_train:
                agent_list[goal].anneal_eps(option_t[goal], option_learned[goal])
        
            metaAgent.anneal_eps(option_meta)
            
            # sync process weights
            sync_process_weights()

            # test network
            if (episodeCount) % (num * 10) == 0:

                data['testMeta/steps'].append(Steps)

                test_external_rew = env.test()
                data['testMeta/external_rew'].append(test_external_rew)
                
                test_external_rew = np.mean(test_external_rew)
                data['testMeta/average_rew'].append(test_external_rew)

                # save the best test weights
                metaAgent.save_weights(test_external_rew)
                
                if test_external_rew >= test_rew:
                    test_rew = test_external_rew

                    save_dict['meta'] = metaAgent.model.state_dict()
                    save_dict['meta_optim'] = metaAgent.optim.state_dict()
                    save_dict['meta_eps'] = metaAgent.eps
                    
                    for goal in goal_to_train:
                        save_dict['goal' + str(goal)] = agent_list[goal].model.state_dict()
                        save_dict['eps' + str(goal)] = agent_list[goal].eps
                        save_dict['goal' + str(goal) + '_optim'] = agent_list[goal].optim.state_dict()
                    torch.save(save_dict, args.logdir + 'better_test_rew.pth')

                    if test_rew > 350:
                        metaAgent.lr_schedule()

                # save test metrics
                logx.add_scalar('testMeta/rew/episode', test_external_rew, episodeCount)
                logx.add_scalar('testMeta/rew/steps', test_external_rew, Steps)

            # logging
            verbose = ''

            for goal in goal_to_train:
                
                if len(option_loss[goal]):

                    v = 'Goal : {0:<2} | eps : {1:.2f} | success_ratio : {2:.2f} | loss : {3:.6f} | option : {4:}\n'.format(
                                                                                                                                    goal,
                                                                                                                                    agent_list[goal].eps,
                                                                                                                                    option_performance[goal],
                                                                                                                                    np.mean(option_loss[goal]),
                                                                                                                                    option_t[goal])
                
                else:
                    v = 'Goal : {0:<2} | eps : {1:.2f} | success_ratio : {2:.2f} | option : {3}\n'.format(
                                                                                                                                    goal,
                                                                                                                                    agent_list[goal].eps,
                                                                                                                                    option_performance[goal],
                                                                                                                                    option_t[goal])
                verbose += v

            if len(meta_loss):

                meta_v = 'Meta : {0:<12} | eps : {1:.2f} | loss : {2:.9f} | option : {3}\n'.format('MetaModel', 
                                                                                                    metaAgent.eps, 
                                                                                                    np.mean(meta_loss),
                                                                                                    option_meta)
            
            else:

                meta_v = 'Meta : {0:<12} | eps : {1:.2f} | option : {2}\n'.format('MetaModel', metaAgent.eps, option_meta)
            
            meta_v += 'ExternalRew : {0:.2f} | AverageExternalRew : {1:.2f} | TestExternalRew : {2:.2f} | Steps : {3}\n'.format( 
                                                                                                                        externalRewards, 
                                                                                                                        cumulative_average_reward,
                                                                                                                        test_external_rew,
                                                                                                                        Steps)
            verbose += meta_v
            
            # print message
            logx.msg('Episode : ' +  str(episodeCount))
            logx.msg(verbose)

            # save metrics
            data['trainGoal/sample_used'].append(option_t)
            data['trainGoal/success_ratio'].append(option_performance)
            data['trainGoal/loss'].append(option_loss)
            data['trainGoal/chosen_times'].append(option_chosen_times)

            data['trainMeta/sample_used'].append(option_meta)
            data['trainMeta/loss'].append(meta_loss)
            data['trainMeta/start_train'] = meta_start_train_episode

            if episodeCount % (num * 10) == 0:
                with open(args.logdir + 'data.pkl', 'wb') as f:
                    pickle.dump(data, f)
            
            t.update(num)

    with open(args.logdir + 'data.pkl', 'wb') as f:
        pickle.dump(data, f)

else:
    for i in range(7):
        agent_list[i].load_weights('./Model/nsrl/', str(i), True)
        t = agent_list[i].eval()
        agent_list[i].set_eps(0)

    metaAgent.load_weights('./Model/nsrl/', 'meta', True)
    metaAgent.eval()
    metaAgent.set_eps(0)
    sync_process_weights()
    test_external_rew = env.test()

    print("test rew for each sub process",test_external_rew)

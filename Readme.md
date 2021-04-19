# Neural Symbolic Reinforcement Learning


- 1.<font color=#0000FF>Introduction</font>[](#Introduction)
- 2.<font color=#0000FF>Dependencies</font>
- 3.<font color=#0000FF>Project Structure</font>
- 4.<font color=#0000FF>Start</font>
- 5.<font color=#0000FF>Applications</font>

## **Introduction**

Neural Symbolic Reinforcement Learning(NSRL) is a Reinforcement Learning framework embedded with Neural Symbolic Logic reasoning. In Montezuma's Revenge, we apply a custom designed detector to transform the state from environment to the symbolic state based on predefined predicate. The meta controller allocates a task and intrinsic reward to the controller, which 
interacts with the environment and collects extrinsic reward. The meta controller consists of three modules, attention module, reasoning module and policy module respectively. The attention module can generate attentions on predicates and paths, whicl would be utilized by the reasoning module to generate the relational paths. After that, the policy module considers all possible relational paths and output the Q value of choosing tasks. The algorithm in the Block World Domain is the same as that in Montezuma's Revenge expect the hierarchial framework. 




## **Dependencies**

Clone this repository and create a virtual python environment. Then install dependenvies using pip

    pip install -r requirements.txt

To visualize Montezuma's Revenge on local machine, the library atari_py need to be recomplied with the setting 'USE SDL' being True.

## **Project Structure**
The whole project is decoupled into the following parts:

- Game : the code for reproduing the result of the Montezuma's Revenge
- Block : the code for reproduing the result of the Block World Domain

## **Start**
This is an example of testing NSRL agent in Montezuma's Revenge

### Montezuma's Revenge

For testing NSRL agent

    python NSRL.py --mode test --model nsrl --num_process 2

This  code would create 2 environments for testing

For training NSRL Agent

    python NSRL.PY --mode train --model nsrl --num_process 8 --logdir ./logdir/nsrl/

This similar code would create 8 environments for sampling data and store the weights of network in the logdir folder

In this projects, we also compare NSRL with another interpretable algorithm SDRL. These two algorithms all perform reasoning on task level. The project SDRL can be found at 
<https://github.com/daomingAU/MontezumaRevenge_SDRL>



### BLock World

For testing NSRL agent

`python main.py --model nsrl --test True --task Stack --logdir ./model/stack/`

This code would test the trained agent on task STACK. Similar setting of parameters can be found in the folder core/argparse.py 

For visualizing the rules in this domain

`python vis.py --model nsrl --task Stack --logdir ./model/stack/`

This code would print the rules at each time step on task STACK




## **Applications**

Right now, we validate NSRL on the Atari_Game Montezuma's Revenge and Block World. We plan to extend NSRL to learn tree-like or junction like rules.

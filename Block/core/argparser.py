import argparse
import time

parser = argparse.ArgumentParser()

# params for training
parser.add_argument("--device", type=str, default='cpu')
parser.add_argument("--num_process", type=int,  default=500)
parser.add_argument("--wait_num", type=int,  default=50)

parser.add_argument("--max_epoch", type=int, default=5000)
parser.add_argument("--step_per_epoch", type=int, default=500)
parser.add_argument("--collect_per_step", type=int, default=10)
parser.add_argument("--repeat_per_collect", type=int, default=5)
parser.add_argument("--episode_per_test", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=64)

parser.add_argument("--model_type", type=str, default='chain')

# params for model
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--gamma", type=float, default=0.99)

# params for symbolic logic
parser.add_argument("--predicate_num", type=int, default=2)
parser.add_argument("--arity_num", type=int, default=8)
parser.add_argument("--path_length", type=int, default=4)
parser.add_argument("--embedding_size", type=int, default=64)

# params for transformer
parser.add_argument("--n_layers", type=int, default=2)
parser.add_argument("--n_head", type=int, default=4)

# params for log
parser.add_argument("--logdir", type=str, default='./logdir/')
parser.add_argument("--task", type=str, default='Stack')
parser.add_argument("--goal_on", type=str, default=False)

parser.add_argument("--test", type=bool, default=False)
parser.add_argument("--test_times", type=int, default=500)

parser.add_argument("--load_model", type=str, default="")

args = parser.parse_args()

if args.task == 'On':
    args.goal_on = True
    args.predicate_num = 3

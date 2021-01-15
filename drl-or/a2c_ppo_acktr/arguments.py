import argparse

import torch


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    # model config
    parser.add_argument('--algo', default='ppo',
                        help='algorithm to use: a2c | ppo')
    parser.add_argument('--lr', type=float, default=2.5e-5,
                        help='learning rate (default: 2.5e-5)')
    parser.add_argument('--eps', type=float, default=1e-5,
                        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument('--alpha', type=float, default=0.99,
                        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--use-gae', action='store_true', default=False,
                        help='use generalized advantage estimation')
    parser.add_argument('--gae-lambda', type=float, default=0.95,
                        help='gae lambda parameter (default: 0.95)')
    parser.add_argument('--entropy-coef', type=float, default=0.01,
                        help='entropy term coefficient (default: 0.01)')
    parser.add_argument('--value-loss-coef', type=float, default=0.5,
                        help='value loss coefficient (default: 0.5)')
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='max norm of gradients (default: 0.5)')
    parser.add_argument('--ppo-epoch', type=int, default=4,
                        help='number of ppo epochs (default: 4)')
    parser.add_argument('--num-mini-batch', type=int, default=32,
                        help='number of batches for ppo (default: 32)')
    parser.add_argument('--clip-param', type=float, default=0.1,
                        help='ppo clip parameter (default: 0.1)')
    parser.add_argument('--recurrent-policy', action='store_true', default=False,
                        help='use a recurrent policy')
    parser.add_argument('--use-linear-lr-decay', action='store_true', default=False,
                        help='use a linear schedule on the learning rate')
    parser.add_argument('--use-linear-clip-decay', action='store_true', default=False,
                        help='use a linear schedule on the ppo clipping parameter')
    
    # running config
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--cuda-deterministic', action='store_true', default=False,
                        help="sets flags for determinism when using CUDA (potentially slow!)")
    parser.add_argument('--num-steps', type=int, default=512,
                        help='number of forward steps (default: 512) i.e.num-step for each update epoch')
    parser.add_argument('--num-pretrain-epochs', type=int, default=30,
                        help='number of pretraining steps  (default: 500)')
    parser.add_argument('--num-pretrain-steps', type=int, default=128,
                        help='number of forward steps for pretraining (default: 128)')
    parser.add_argument('--ckpt-steps', type=int, default=10000,
                        help='number of iteration steps for each checkpoint when training')
    parser.add_argument('--num-env-steps', type=int, default=10000000,
                        help='number of environment steps to train (default: 1000000)')
    parser.add_argument('--env-name', default='Abi',
                        help='environment to train on (default: Abi)') #temporarily deprecated
    parser.add_argument('--log-dir', default='/tmp/DRL-OR',
                        help='directory to save agent logs (default: /tmp/DRL-OR)')
    parser.add_argument('--save-dir', default='./trained_models/',
                        help='directory to save agent logs (default: ./trained_models/)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument("--demand-matrix", default='test.txt', 
                        help='demand matrix input file name (default:test.txt)')
    parser.add_argument("--model-load-path", default=None,
                        help='load model parameters from the model-load-path')
    parser.add_argument("--model-save-path", default=None,
                        help='save model parameters at the model-save-path')
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    assert args.algo in ['a2c', 'ppo']
    
    return args

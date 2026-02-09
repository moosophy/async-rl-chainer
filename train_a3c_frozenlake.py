import argparse

import chainer
from chainer import links as L
from chainer import functions as F
from chainer import optimizers
import numpy as np

import policy
import v_function
import a3c
import random_seed
import rmsprop_async
from init_like_torch import init_like_torch
import run_a3c

import time
from datetime import datetime


N_OBS = 16  # FrozenLake-v1 4x4 grid: 16 discrete states
N_ACTIONS = 4  # left, down, right, up


def phi(obs):
    """One-hot encode the discrete observation."""
    one_hot = np.zeros(N_OBS, dtype=np.float32)
    one_hot[int(obs)] = 1.0
    return one_hot


class FrozenLakeWrapper:
    """Wraps gymnasium FrozenLake to match the old gym API expected by run_a3c."""

    def __init__(self, is_slippery=True):
        import gymnasium as gym
        self.env = gym.make('FrozenLake-v1', is_slippery=is_slippery)

    def reset(self):
        obs, _ = self.env.reset()
        return obs

    def step(self, action):
        obs, r, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return obs, r, done, info


class A3CFFSimple(chainer.ChainList, a3c.A3CModel):

    def __init__(self, n_obs, n_actions, n_hidden=64):
        self.fc = L.Linear(n_obs, n_hidden)
        self.pi = policy.FCSoftmaxPolicy(n_hidden, n_actions)
        self.v = v_function.FCVFunction(n_hidden)
        super().__init__(self.fc, self.pi, self.v)
        init_like_torch(self)

    def pi_and_v(self, state, keep_same_state=False):
        out = F.relu(self.fc(state))
        return self.pi(out), self.v(out)


class A3CLSTMSimple(chainer.ChainList, a3c.A3CModel):

    def __init__(self, n_obs, n_actions, n_hidden=64):
        self.fc = L.Linear(n_obs, n_hidden)
        self.lstm = L.LSTM(n_hidden, n_hidden)
        self.pi = policy.FCSoftmaxPolicy(n_hidden, n_actions)
        self.v = v_function.FCVFunction(n_hidden)
        super().__init__(self.fc, self.lstm, self.pi, self.v)
        init_like_torch(self)

    def pi_and_v(self, state, keep_same_state=False):
        out = F.relu(self.fc(state))
        if keep_same_state:
            prev_h, prev_c = self.lstm.h, self.lstm.c
            out = self.lstm(out)
            self.lstm.h, self.lstm.c = prev_h, prev_c
        else:
            out = self.lstm(out)
        return self.pi(out), self.v(out)

    def reset_state(self):
        self.lstm.reset_state()

    def unchain_backward(self):
        self.lstm.h.unchain_backward()
        self.lstm.c.unchain_backward()


def main():
    import logging
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                       datefmt='%Y-%m-%d %H:%M:%S')

    parser = argparse.ArgumentParser()
    parser.add_argument('processes', type=int)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--outdir', type=str, default=None)
    parser.add_argument('--t-max', type=int, default=20)
    parser.add_argument('--beta', type=float, default=1e-2)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--steps', type=int, default=5 * 10 ** 6)
    parser.add_argument('--lr', type=float, default=7e-4)
    parser.add_argument('--eval-frequency', type=int, default=10 ** 4)
    parser.add_argument('--eval-n-runs', type=int, default=10)
    parser.add_argument('--use-lstm', action='store_true')
    parser.set_defaults(use_lstm=False)
    args = parser.parse_args()

    print('Arguments:', args.use_lstm)

    if args.seed is not None:
        random_seed.set_random_seed(args.seed)

    def make_env(process_idx, test):
        return FrozenLakeWrapper(is_slippery=False)

    def model_opt():
        if args.use_lstm:
            print("Using LSTM model")
            model = A3CLSTMSimple(N_OBS, N_ACTIONS)
        else:
            print("Using Feed-Forward model")
            model = A3CFFSimple(N_OBS, N_ACTIONS)
        opt = chainer.optimizers.RMSprop(lr=args.lr, eps=1e-1, alpha=0.99)
        opt.setup(model)
        opt.add_hook(chainer.optimizer.GradientClipping(40))
        return model, opt

    run_a3c.run_a3c(args.processes, make_env, model_opt, phi, t_max=args.t_max,
                    beta=args.beta, profile=args.profile, steps=args.steps,
                    eval_frequency=args.eval_frequency,
                    args=args)


if __name__ == '__main__':
    import sys
    start_time = time.time()
    start_timestamp = datetime.utcfromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')
    n_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    print(f'[BENCHMARK] start_time:{start_timestamp} workers:{n_workers} env:FrozenLake-v1')
    main()
    end_time = time.time()
    end_timestamp = datetime.utcfromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')
    total_duration = end_time - start_time
    print(f'[BENCHMARK] end_time:{end_timestamp} workers:{n_workers} total_duration:{total_duration:.2f}s')

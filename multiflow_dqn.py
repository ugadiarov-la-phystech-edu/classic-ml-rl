import argparse
import collections
import copy
import importlib
import math
import os
import pickle
import traceback

import gym
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import numpy as np
import torch

import wandb

from envs import catch, boulder, roadrunner, study, memory_corridor, tamagotchi, golf, supermarket, trashbot
from envs.wrappers import FlattenWrapper, StepWrapper, FrameStackWrapper

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
device = 'cpu'


class QCritic:
    def __init__(self, cls, n_actions, **model_kwargs):
        self.cls = cls
        self.kwargs = model_kwargs
        self.n_actions = n_actions
        self.model = None

    def q_value(self, state_action):
        return self.model.predict(state_action)

    def _q_values(self, state):
        state_action = np.stack(
            [np.concatenate((state, np.full((state.shape[0], 1), fill_value=action, dtype=state.dtype)), axis=-1) for
             action in range(self.n_actions)], axis=1)
        return self.q_value(state_action.reshape((-1, state_action.shape[-1]))).reshape(
            (state.shape[0], self.n_actions))

    def act(self, state):
        return np.argmax(self._q_values(state), axis=-1)

    def value(self, state):
        return np.max(self._q_values(state), axis=-1)

    def train(self, state_action, q_value, init=False):
        if init:
            assert self.model is None
            self.model = self.cls(**self.kwargs)
            return self.model.fit(state_action, q_value)
        else:
            return self.model.partial_fit(state_action, q_value)


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def get_data(self):
        return list(self.memory)

    def __len__(self):
        return len(self.memory)


def plot_metric(metric_history, show_result=False):
    plt.figure(1)
    metric_history_tensor = torch.tensor(metric_history, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(metric_history_tensor.numpy())
    # Take 100 episode averages and plot them too
    if len(metric_history_tensor) >= 100:
        means = metric_history_tensor.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


def select_action(state, q_critic, exploration_epsilon):
    sample = random.random()
    if sample > exploration_epsilon and q_critic.model is not None:
        action = q_critic.act(state)[np.newaxis]
    else:
        action = np.full((1, 1), fill_value=env.action_space.sample(), dtype=np.int64)

    return torch.as_tensor(action, device=device, dtype=torch.int64)


def optimize_model(q_critic, target_q_critic, buffer, batch_size, gamma, init=False):
    if init:
        transitions = buffer.get_data()
    else:
        transitions = buffer.sample(batch_size)
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool).numpy()
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None]).numpy()
    state_batch = torch.cat(batch.state).numpy()
    action_batch = torch.cat(batch.action).numpy()
    reward_batch = torch.cat(batch.reward).numpy()

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = np.zeros((state_batch.shape[0],), dtype=np.float32)
    if target_q_critic.model is not None:
        next_state_values[non_final_mask] = target_q_critic.value(non_final_next_states)
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * gamma) + reward_batch
    q_critic.train(np.concatenate([state_batch, action_batch], axis=-1), expected_state_action_values, init=init)


def epsilon_schedule(eps_start, eps_end, eps_decay_steps, step):
    if step >= eps_decay_steps:
        return eps_end

    return eps_start + (eps_end - eps_start) * step / eps_decay_steps


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cls', type=str, required=True)
    parser.add_argument('--env_id', required=True, type=str)
    parser.add_argument('--exploration_eps_start', type=float, default=1.0)
    parser.add_argument('--exploration_eps_end', type=float, default=0.05)
    parser.add_argument('--exploration_decay_steps', type=int, default=5000)
    parser.add_argument('--save_every_steps', type=int, default=5000)
    parser.add_argument('--log_every_steps', type=int, default=1000)
    parser.add_argument('--wandb_project', type=str, required=True)
    parser.add_argument('--wandb_group', type=str, required=True)
    parser.add_argument('--wandb_run_name', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--buffer_size', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--target_update_interval', type=int, default=100)
    parser.add_argument('--frame_stack', type=int, default=1)
    parser.add_argument('--training_starts', type=int, default=500)
    parser.add_argument('--train_freq', type=int, default=1)
    parser.add_argument('--optimization_steps', type=int, default=1)
    parser.add_argument('--reinit_interval', type=int, default=math.inf)
    args = parser.parse_args()
    cls = None
    for module in ["skmultiflow.trees", "skmultiflow.lazy", "skmultiflow.meta"]:
        try:
            cls = getattr(importlib.import_module(module), args.cls)
            break
        except Exception:
            print(traceback.format_exc())

    if cls is None:
        raise ValueError(f'Cannot find class by name: {args.cls}')

    batch_size = args.batch_size
    gamma = 0.99
    target_update_interval = args.target_update_interval
    reinit_interval = args.reinit_interval
    buffer_size = args.buffer_size
    training_starts = args.training_starts
    train_freq = args.train_freq
    optimization_steps = args.optimization_steps

    env_id = args.env_id
    env = FlattenWrapper(gym.make(env_id))
    if args.frame_stack > 1:
        env = FrameStackWrapper(env, n_frames=args.frame_stack)

    env = StepWrapper(env)

    # Get number of actions from gym action space
    n_actions = env.action_space.n
    # Get the number of state observations
    state = env.reset()

    # kwargs = {'grace_period': BATCH_SIZE} if 'Hoeffding' in args.cls else {}
    kwargs = {}
    if 'RandomForest' in args.cls:
        kwargs['n_estimators'] = 100

    q_critic = QCritic(cls, n_actions, **kwargs)
    target_q_critic = q_critic
    memory = ReplayMemory(buffer_size)

    steps_done = 0
    num_episodes = 0
    next_save_step = args.save_every_steps
    next_log_step = args.log_every_steps
    episode_durations = collections.deque(maxlen=100)
    episode_return_history = collections.deque(maxlen=100)
    run = wandb.init(project=args.wandb_project, group=args.wandb_group, name=args.wandb_run_name, config=kwargs,
                     sync_tensorboard=True, monitor_gym=True, )

    init = True
    while True:
        # Initialize the environment and get its state
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        episode_return = 0
        for t in count():
            epsilon = epsilon_schedule(args.exploration_eps_start, args.exploration_eps_end,
                                       args.exploration_decay_steps, steps_done)
            action = select_action(state.numpy(), q_critic, epsilon)
            observation, reward, done, _ = env.step(action.item())
            steps_done += 1
            episode_return += reward
            reward = torch.tensor([reward], device=device)
            if done:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # Store the transition in memory
            memory.push(state, action.to(torch.float32), next_state, reward)

            # Move to the next state
            state = next_state

            if steps_done >= training_starts and steps_done % train_freq == 0:
                # Perform optimization
                for _ in range(optimization_steps):
                    optimize_model(q_critic, target_q_critic, memory, batch_size, gamma, init=init)
                    init = False

                if target_q_critic.model is None:
                    target_q_critic = copy.deepcopy(q_critic)

            if steps_done % target_update_interval == 0:
                target_q_critic = copy.deepcopy(q_critic)

            if steps_done >= next_save_step:
                next_save_step += args.save_every_steps
                with open(os.path.join(args.save_path, 'model.pkl'), 'wb') as file_obj:
                    pickle.dump(q_critic, file_obj)

            if steps_done % reinit_interval == 0:
                q_critic = QCritic(cls, n_actions, **kwargs)
                optimize_model(q_critic, target_q_critic, memory, batch_size, gamma, init=True)


            if steps_done >= next_log_step:
                next_log_step += args.log_every_steps
                wandb.log({'global_step': steps_done, 'rollout/ep_len_mean': np.mean(episode_durations),
                           'rollout/ep_rew_mean': np.mean(episode_return_history), 'time/episodes': num_episodes})

            if done:
                episode_return_history.append(episode_return)
                episode_return = 0
                num_episodes += 1
                episode_durations.append(t + 1)
                break

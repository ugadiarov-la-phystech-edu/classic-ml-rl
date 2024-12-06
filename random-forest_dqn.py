import argparse
import copy

import gym
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import numpy as np
import torch

import wandb

from sklearn.ensemble import RandomForestRegressor

import catch


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


class FlattenWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=env.observation_space.low.reshape(-1),
                                                high=env.observation_space.high.reshape(-1),
                                                shape=(np.prod(np.array(env.observation_space.shape)),),
                                                dtype=env.observation_space.dtype)

    def observation(self, observation):
        return observation.reshape(-1)


class RFQ:
    def __init__(self, n_actions, **rf_kwargs):
        self.kwargs = rf_kwargs
        self.n_actions = n_actions
        self.rf_regressor = None

    def q_value(self, state_action):
        return self.rf_regressor.predict(state_action)

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

    def train(self, state_action, q_value):
        self.rf_regressor = RandomForestRegressor(**self.kwargs)
        return self.rf_regressor.fit(state_action, q_value)


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


def select_action(state, rfq, steps_done, total_timesteps, eps_start, eps_end, exploration_fraction):
    sample = random.random()
    eps_threshold = eps_end
    if steps_done <= exploration_fraction * total_timesteps:
        eps_threshold = eps_start - steps_done / total_timesteps / exploration_fraction * (eps_start - eps_end)

    if sample > eps_threshold and rfq.rf_regressor is not None:
        action = rfq.act(state)[np.newaxis]
    else:
        action = np.full((1, 1), fill_value=env.action_space.sample(), dtype=np.int64)

    return torch.as_tensor(action, device=device, dtype=torch.int64)


def optimize_model(rfq, target_rfq, buffer, batch_size, gamma):
    transitions = buffer.get_data()
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
    if target_rfq.rf_regressor is not None:
        next_state_values[non_final_mask] = target_rfq.value(non_final_next_states)
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * gamma) + reward_batch
    rfq.train(np.concatenate([state_batch, action_batch], axis=-1), expected_state_action_values)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_id', required=True, type=str)
    args = parser.parse_args()

    # CartPole-v1
    # BATCH_SIZE = 128
    # GAMMA = 0.99
    # EPS_START = 0.9
    # EPS_END = 0.05
    # EXPLORATION_FRACTION = 1.0
    # TAU = 0.005
    # LR = 1e-4
    # BUFFER_SIZE = 10000
    # TRAINING_STARTS = 5000
    # TOTAL_TIMESTEPS = 20000
    # TRAIN_FREQ = 1

    # env = gym.make("CartPole-v1")


    BATCH_SIZE = 32
    GAMMA = 0.99
    EPS_START = 1
    EPS_END = 0.05
    EXPLORATION_FRACTION = 0.35
    TARGET_UPDATE_INTERVAL = 100
    LR = 1e-3
    BUFFER_SIZE = 1000
    TRAINING_STARTS = 500
    TOTAL_TIMESTEPS = 20000
    TRAIN_FREQ = 1

    env_id = args.env_id
    env = FlattenWrapper(gym.make(env_id))

    # Get number of actions from gym action space
    n_actions = env.action_space.n
    # Get the number of state observations
    state = env.reset()

    rfq = RFQ(n_actions, n_jobs=-1)
    target_rfq = rfq
    memory = ReplayMemory(BUFFER_SIZE)

    steps_done = 0
    num_episodes = 0
    episode_durations = []
    episode_return_history = []

    run = wandb.init(project=env_id, group='RandomForest', name='run-0', sync_tensorboard=True, monitor_gym=True,)

    while steps_done < TOTAL_TIMESTEPS:
        # Initialize the environment and get its state
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        episode_return = 0
        for t in count():
            action = select_action(state.numpy(), rfq, steps_done, TOTAL_TIMESTEPS, EPS_START, EPS_END, EXPLORATION_FRACTION)
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

            if steps_done > TRAINING_STARTS and steps_done % TRAIN_FREQ == 0:
                # Perform one step of the optimization (on the policy network)
                optimize_model(rfq, target_rfq, memory, BATCH_SIZE, GAMMA)
                if target_rfq.rf_regressor is None:
                    target_rfq = copy.deepcopy(rfq)

            if steps_done % TARGET_UPDATE_INTERVAL == 0:
                target_rfq = copy.deepcopy(rfq)

            if done:
                episode_return_history.append(episode_return)
                episode_return = 0
                num_episodes += 1
                episode_durations.append(t + 1)
                # plot_metric(episode_durations)
                plot_metric(episode_return_history)
                wandb.log({'global_step': steps_done, 'rollout/ep_len_mean': np.mean(episode_durations[-100:]),
                           'rollout/ep_rew_mean': np.mean(episode_return_history[-100:]), 'time/episodes': num_episodes})
                break

    print('Complete')
    # plot_metric(episode_durations, show_result=True)
    # plot_metric(episode_return_history, show_result=True)
    # plt.ioff()
    # plt.show()

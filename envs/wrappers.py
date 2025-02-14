import gym
import numpy as np


class StepWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        step_result = super().step(action)
        if len(step_result) == 4:
            return step_result
        elif len(step_result) == 5:
            observation, reward, terminated, truncated, info = step_result
            done = terminated or truncated
            info['TimeLimit.truncated'] = truncated

            return observation, reward, done, info
        else:
            raise ValueError(f'The output of "step" function must contain 4 or 5 elements, actual: {len(step_result)}')



class FlattenWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        observation_space = env.observation_space
        if isinstance(observation_space, gym.spaces.Discrete):
            self.shape = (1,)
            self.observation_space = gym.spaces.Box(low=0, high=observation_space.n, shape=self.shape, dtype=observation_space.dtype)
        elif isinstance(env.observation_space, gym.spaces.MultiDiscrete):
            self.shape = env.observation_space.shape
            self.observation_space = gym.spaces.Box(low=0, high=observation_space.nvec.max(), shape=self.shape, dtype=observation_space.dtype)
        elif isinstance(env.observation_space, gym.spaces.Box):
            self.shape = (np.prod(np.array(env.observation_space.shape)),)
            self.observation_space = gym.spaces.Box(low=env.observation_space.low.reshape(-1),
                                                    high=env.observation_space.high.reshape(-1),
                                                    shape=(np.prod(np.array(env.observation_space.shape)),),
                                                    dtype=env.observation_space.dtype)
        else:
            raise ValueError(f'Unexpected space: {env.observation_space}')

    def observation(self, observation):
        return np.asarray(observation).reshape(self.shape)


class FrameStackWrapper(gym.Wrapper):
    def __init__(self, env, n_frames):
        super(FrameStackWrapper, self).__init__(env)
        self.n_frames = n_frames
        observation_space = env.observation_space
        low = np.stack([observation_space.low] * self.n_frames)
        high = np.stack([observation_space.high] * self.n_frames)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=observation_space.dtype)
        self.frames = np.zeros((self.n_frames), dtype=np.int32)

    def reset(self):
        obs = self.env.reset()

        # Set observation to num_doors+1 if no observation is made yet (at the start of the rollout)
        self.frames[...] = self.observation_space.high.max()
        self.frames[-1] = obs
        return self.frames.flatten()

    def step(self, action):
        obs, reward, terminated, info = self.env.step(action)
        self.frames[:-1] = self.frames[1:]
        self.frames[-1] = obs
        return self.frames.flatten(), reward, terminated, info

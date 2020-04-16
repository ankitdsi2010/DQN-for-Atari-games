
import gym
import atari_py
import numpy as np
from collections import deque
from gym.spaces.box import Box
import cv2
import random

def make_env(env_id, env_conf):
    env = gym.make(env_id)
    
    if 'NoFrameSkip' in env_id:
        assert 'NoFrameSkip' in env.spec.id
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=env_conf['skip_rate'])
        
    if env_conf['episodic_life']:
        env = EpisodicLifeEnv(env)
        
    try:
        if 'FIRE' in env.wrapped.get_action_meanings():
            env = FireResetEnv(env)
    except AttributeError:
        pass
    
    env = AtariRescale(env, env_conf['useful_region'])
    
    if env_conf['normalize_observation']:
        env = NormalizedEnv(env)
        
    env = FrameStack(env, env_conf['num_frames_to_stack'])
    
    return env

def get_games_list():
    return atari_py.list_games()

class ClipRewardEnv(gym.RewardWrapper):
    
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        
    def reward(self, reward):
        return np.sign(reward)
    
def process_frame_84(frame, conf):
    frame = frame[conf["crop1"]:conf["crop2"] + 160, :160]
    frame = frame.mean(2)
    frame = cv2.resize(frame, (84, conf["dimension2"]))
    frame = cv2.resize(frame, (84, 84))
    frame = np.reshape(frame, [1, 84, 84])
    return frame

class AtariRescale(gym.ObservationWrapper):
    
    def __init__(self, env, env_conf):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = Box(0, 255, [1, 84, 84], dtype=np.uint8)
        self.conf = env_conf
        
    def observation(self, observation):
        return process_frame_84(observation, self.conf)
    
class NormalizedEnv(gym.ObservationWrapper):
    
    def __init__(self, env=None):
        gym.ObservationWrapper.__init__(self, env)
        self.state_mean = 0
        self.state_std = 0
        self.alpha = 0.9999
        self.num_steps = 0
        
    def observation(self, observation):
        self.num_steps += 1
        self.state_mean = self.state_mean * self.alpha + \
            observation.mean() * (1 - self.alpha)
        self.state_std = self.state_std * self.alpha + \
            observation.std() * (1 - self.alpha)
            
        unbiased_mean = self.state_mean / (1 - pow(self.alpha, self.num_steps))
        unbiased_std = self.state_std / (1 - pow(self.alpha, self.num_steps))
        
        return (observation - unbiased_mean) / (unbiased_std + 1e-8)
    
class NoopResetEnv(gym.Wrapper):
    
    def __init__(self, env, noop_max=30):
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'
        
    def reset(self):
        self.env.reset()
        noops = random.randrange(1, self.noop_max+1)
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
        return obs
    
    def step(self, action):
        return self.env.step(action)
    
class FireResetEnv(gym.Wrapper):
    
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3
        
    def reset(self):
        self.env.reset()
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset()
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset()
        return obs
    
    def step(self, action):
        return self.env.step(action)
    
class EpisodicLifeEnv(gym.Wrapper):
    
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True
        
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = True
        lives = info['ale.lives']
        if lives < self.lives and lives > 0:
            done = True
            self.was_real_done = False
        self.lives = lives
        return obs, reward, done, info
    
    def reset(self):
        if self.was_real_done:
            obs = self.env.reset()
            self.lives  = 0
        else:
            obs, _, _, info = self.env.step(0)
            self.lives = info['ale.lives']
        return obs
    
class MaxAndSkipEnv(gym.Wrapper):
    
    def __init__(self, env=None, skip=4):
        gym.Wrapper.__init__(self, env)
        self._obs_buffer = deque(maxlen=2)
        self._skip = skip
        
    def step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
            
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, done, info
    
    def reset(self):
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs
    
class FrameStack(gym.Wrapper):
    
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = Box(low=0, high=255, shape=(shp[0] * k, shp[1], shp[2]), dtype=np.uint8)
        
    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()
    
    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info
    
    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))
    
class LazyFrames(object):
    
    def __init__(self, frames):
        self._frames = frames
        self._out = None
        
    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=0)
            self._frames = None
        return self._out
    
    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out
    
    def __len__(self):
        return len(self._force())
    
    def __getitem__(self, i):
        return self._force()[i]
    
    
    
    
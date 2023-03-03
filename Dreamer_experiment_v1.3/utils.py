import math
import numpy as np
import torch
import random 
from collections import namedtuple, deque
from typing import Optional, Tuple

class TransitionBuffer():
    """
        Replay buffer for training with RNN
        reference:https://github.com/RajGhugare19/dreamerv2
    """
    def __init__(self, capacity, obs_shape: Tuple[int], action_size: int,
            seq_len: int,  batch_size: int, obs_type=np.uint8, action_type=np.float32):

        self.capacity = capacity
        self.obs_shape = obs_shape
        self.action_size = action_size
        self.obs_type = obs_type
        self.action_type = action_type
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.idx = 0
        self.full = False
        self.observation = np.empty((capacity, *obs_shape), dtype=obs_type) 
        self.action = np.empty((capacity, action_size), dtype=np.float32)
        self.reward = np.empty((capacity,), dtype=np.float32) 
        self.terminal = np.empty((capacity,), dtype=bool)

    def add( self, obs: np.ndarray, action: np.ndarray, reward: float, done: bool):
        self.observation[self.idx] = obs
        self.action[self.idx] = action 
        self.reward[self.idx] = reward
        self.terminal[self.idx] = done
        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def _sample_idx(self, L):
        valid_idx = False
        while not valid_idx:
            idx = np.random.randint(0, self.capacity if self.full else self.idx - L)
            idxs = np.arange(idx, idx + L) % self.capacity
            valid_idx = not self.idx in idxs[1:] 
        # print(idxs)
        return idxs

    def _retrieve_batch(self, idxs, n, l):
        vec_idxs = idxs.transpose().reshape(-1)
        observation = self.observation[vec_idxs]
        
        return observation.reshape(l, n, *self.obs_shape), self.action[vec_idxs].reshape(l, n, -1), self.reward[vec_idxs].reshape(l, n), self.terminal[vec_idxs].reshape(l, n)

    def sample(self):
        n = self.batch_size
        l = self.seq_len+1
        obs, act, rew, term = self._retrieve_batch(np.asarray([self._sample_idx(l) for _ in range(n)]), n, l)
        obs, act, rew, term = self._shift_sequences(obs,act,rew,term)

        return obs, act, rew, term
    
    # t-1, t, t+1, ..., t+L
    def _shift_sequences(self, obs, actions, rewards, terminals):
        obs = obs[1:]
        actions = actions[:-1]
        rewards = rewards[:-1]
        terminals = terminals[:-1]

        return obs, actions, rewards, terminals

    def save_buffer(self, directory, episode):
        obs = self.observation
        action = self.action
        reward = self.reward
        terminal = self.terminal

        np.save("{}/Save_Buffer/epi={}_idx={}_obs".format(directory, episode, self.idx), obs)
        np.save("{}/Save_Buffer/epi={}_idx={}_action".format(directory, episode, self.idx), action)
        np.save("{}/Save_Buffer/epi={}_idx={}_reward".format(directory, episode, self.idx), reward)
        np.save("{}/Save_Buffer/epi={}_idx={}_terminal".format(directory, episode, self.idx), terminal)
        print('Saved Replay Buffer')

    def load_buffer(self, directory, episode, idx):
        self.observation = np.load("{}/Save_Buffer/epi={}_idx={}_obs.npy".format(directory, episode, idx))
        self.action = np.load("{}/Save_Buffer/epi={}_idx={}_action.npy".format(directory, episode, idx))
        self.reward = np.load("{}/Save_Buffer/epi={}_idx={}_reward.npy".format(directory, episode, idx))
        self.terminal = np.load("{}/Save_Buffer/epi={}_idx={}_terminal.npy".format(directory, episode, idx))

        self.idx = idx

        print("Loaded Replay Buffer")
        print("episode: ", episode," index: ", self.idx)


# 各アクチュエータの行動をaction_outputにまとめる
def define_action_to_be_selected(action_first, action_second):
    action_output = []
    for i in range(len(action_first)):
        for j in range(len(action_second)):
            action_output.append([action_first[i], action_second[j]])

    return action_output

def preprocess_obs(obs):
    """
        conbert image from [0, 255] to [-0.5, 0.5]
    """
    obs = obs.astype(np.float32)
    normalized_obs = obs / 255.0 - 0.5

    return normalized_obs


def preprocess_reward(reward):
    """
        transform np.tanh()
    """
    reward = reward.astype(np.float32)
    transformed_reward = np.tanh(reward)

    return transformed_reward


def compute_return(
                reward: torch.Tensor,
                value: torch.Tensor,
                discount: torch.Tensor,
                bootstrap: torch.Tensor,
                lambda_: float
            ):
    """
        Compute the discounted reward for a batch of data.
        reward, value, and discount are all shape [horizon - 1, batch, 1] (last element is cut off)
        Bootstrap is [batch, 1]
        reference:https://github.com/RajGhugare19/dreamerv2
    """
    next_values = torch.cat([value[1:], bootstrap[None]], 0) ### t+1 ~ t+H
    target = reward + discount * next_values * (1 - lambda_)
    timesteps = list(range(reward.shape[0] - 1, -1, -1)) ### array=H-2 ~ 0
    outputs = []
    accumulated_reward = bootstrap
    for t in timesteps:
        inp = target[t]
        discount_factor = discount[t]
        accumulated_reward = inp + discount_factor * lambda_ * accumulated_reward
        outputs.append(accumulated_reward)
    returns = torch.flip(torch.stack(outputs), [0])

    return returns


def save_eval_numpy(directory, hsc_image, recon_image, episode, exp_info):
    np.save("{}/Prediction/epi={}_hsc_numpy_{}".format(directory, episode, exp_info), hsc_image)
    print('Saved hsc_numpy')

    np.save("{}/Prediction/epi={}_recon_numpy_{}".format(directory, episode, exp_info), recon_image)
    print('Saved recon_numpy')

def save_openl_numpy(directory, openl_image, episode, exp_info):
    np.save("{}/Prediction/epi={}_openloop_numpy_{}".format(directory, episode, exp_info), openl_image)
    print('Saved openl_numpy')

def save_fullyopenl_numpy(directory, openl_image, episode, exp_info):
    np.save("{}/Prediction/epi={}_fullyopenloop_numpy_{}".format(directory, episode, exp_info), openl_image)
    print('Saved openl_numpy')
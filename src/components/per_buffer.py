from typing import DefaultDict
from sympy import EX
import torch as th
import numpy as np
from types import SimpleNamespace as SN
from .episode_buffer import EpisodeBatch

class PERBuffer(EpisodeBatch):
    """Implements non-uniform sampling from the episode buffer. Weighted proportionally based on episode return.
    """
    def __init__(self, scheme, groups, buffer_size, max_seq_length, per_alpha, per_epsilon, preprocess=None, device="cpu"):
        super(PERBuffer, self).__init__(scheme, groups, buffer_size, max_seq_length, preprocess=preprocess, device=device)
        self.buffer_size = buffer_size  # same as self.batch_size but more explicit
        self.buffer_index = 0
        self.episodes_in_buffer = 0
        
        self.per_alpha = per_alpha
        self.per_epsilon = per_epsilon
        self.pvalues = {}
        self.max_reward_sum = 0.0
        self.reward_sum = {}
        self.e_sampled = DefaultDict(lambda : False)

    def insert_episode_batch(self, ep_batch):
        """Insert episode into replay buffer.

        Args:
            ep_batch (EpiosdeBatch): Episode to be inserted
        """
        print(f'inserting episode batch, buffer idx {self.buffer_index}, ep batch size {ep_batch.batch_size}')
        if self.buffer_index + ep_batch.batch_size <= self.buffer_size:  
            
            self.reward_sum[self.buffer_index] = (th.sum(ep_batch["reward"][:, :-1]) + self.per_epsilon)**self.per_alpha
            if self.reward_sum[self.buffer_index] > self.max_reward_sum:
                self.max_reward_sum = self.reward_sum[self.buffer_index]
            self.pvalues[self.buffer_index] = self.max_reward_sum
            
            self.update(ep_batch.data.transition_data,
                        slice(self.buffer_index, self.buffer_index + ep_batch.batch_size),
                        slice(0, ep_batch.max_seq_length),
                        mark_filled=False)
            self.update(ep_batch.data.episode_data,
                        slice(self.buffer_index, self.buffer_index + ep_batch.batch_size))
            self.buffer_index = (self.buffer_index + ep_batch.batch_size)
            self.episodes_in_buffer = max(self.episodes_in_buffer, self.buffer_index)
            self.buffer_index = self.buffer_index % self.buffer_size  # resets buffer index once it is greater than buffer size, allows it to then remove oldest epsiodes
            assert self.buffer_index < self.buffer_size
            
        else: 
            buffer_left = self.buffer_size - self.buffer_index  # i guess this is for when buffer_size % batch_size > 0
            print(f' -- Uneaven entry to buffer -- ')
            self.insert_episode_batch(ep_batch[0:buffer_left, :])
            self.insert_episode_batch(ep_batch[buffer_left:, :])

    def can_sample(self, batch_size):
        return self.episodes_in_buffer >= batch_size

    def sample(self, batch_size):
        """Returns a sample of episodes from the replay buffer

        Args:
            batch_size (int): Number of episodes to return
        """
        assert self.can_sample(batch_size)
        if self.episodes_in_buffer == batch_size:
            return self[:batch_size]
        else:
            probs = np.fromiter(self.pvalues.values(), dtype=float)
            probs = probs/np.sum(probs)
            ep_ids = np.random.choice(self.episodes_in_buffer, batch_size, replace=False)
            for i in ep_ids:
                if not self.e_sampled[i]:
                    self.pvalues[i] = self.reward_sum[i]
                    self.e_sampled[i] = True
            return self[ep_ids]

    def __repr__(self):
        return "ReplayBuffer. {}/{} episodes. Keys:{} Groups:{}".format(self.episodes_in_buffer,
                                                                        self.buffer_size,
                                                                        self.scheme.keys(),
                                                                        self.groups.keys())


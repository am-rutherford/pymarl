import pathlib
from copy import deepcopy
from math import floor
from typing import DefaultDict
from sympy import EX
import torch as th
import numpy as np
from types import SimpleNamespace as SN
from .episode_buffer import EpisodeBatch
from .epsilon_schedules import RiseThenFlatSchedule

class PERBuffer(EpisodeBatch):
    """Implements non-uniform sampling from the episode buffer. Weighted proportionally based on episode return.
    """
    def __init__(self, args, scheme, groups, buffer_size, max_seq_length, preprocess=None, device="cpu"):
        """ 
        Args:
            per_alpha: Exponent applied to the sum of the reward score and per_epsilon. Must lie in the range [0, 1].
            per_epsilon: Constant added to reward score.
            per_beta: importance sampling exponent, controls how much prioritization to apply. Must lie in the range [0, 1].
        """
        super(PERBuffer, self).__init__(scheme, groups, buffer_size, max_seq_length, preprocess=preprocess, device=device)
        self.buffer_size = buffer_size  # same as self.batch_size but more explicit
        self.buffer_index = 0
        self.episodes_in_buffer = 0
        self.device = args.device
        
        assert (args.per_alpha >= 0) and (args.per_alpha <= 1), "per_alpha is out of bounds, must lie in the range [0, 1]"
        assert args.per_epsilon >= 0, "per_epsilon must be positive"
        assert (args.per_beta >= 0) and (args.per_beta <= 1), "per_beta is out of bounds, must lie in the range [0, 1]"
        assert (args.per_beta_anneal >= 0) and (args.per_beta_anneal <= 1), "per_beta_anneal is out of bounds, must lie in the range [0, 1]"
        
        self.per_alpha = args.per_alpha
        self.per_epsilon = args.per_epsilon
        self.per_beta_schedule = RiseThenFlatSchedule(args.per_beta, 1, floor(args.t_max * args.per_beta_anneal), decay="linear")
        self.per_beta = self.per_beta_schedule.eval(0)
        
        print(f'Initialising PER buffer, annealing beta from {args.per_beta} to 1 over {floor(args.t_max * args.per_beta_anneal)} timesteps.')
        
        self.use_offset = True if args.env_args["reward_type"] == "time-cost" else False
        
        self.offset = 0.0
        self.origin_reward_idx = 0
        self.og_reward = th.zeros((buffer_size, 1, 1), device=self.device)
        
        self.pvalues = th.zeros((buffer_size, 1, 1), device=self.device)
        self.max_reward_sum = 0.0
        self.max_reward_idx = 0
        self.reward_sum = th.zeros((buffer_size, 1, 1), device=self.device)
        self.e_sampled = th.zeros((buffer_size, 1, 1), device=self.device)
        
        # for logging values
        self.buffer_counter = 0
        self.reward_sum_record = {}
        self.sample_count = {}

    def insert_episode_batch(self, ep_batch):
        """Insert episode into replay buffer.

        Args:
            ep_batch (EpiosdeBatch): Episode to be inserted
        """
        #print(f'inserting episode batch, buffer idx {self.buffer_index}, ep batch size {ep_batch.batch_size}')
        if self.buffer_index + ep_batch.batch_size <= self.buffer_size:  
            ## PER values
            assert ep_batch.batch_size == 1
            reward = th.sum(ep_batch["reward"])
            
            if self.use_offset:
                if reward < -1*self.offset: # reward is lower than any currently in buffer - shift origin
                    self.og_reward = self.og_reward - (self.offset + reward)#.to(self.device) -- try fix cuda implementation at some point
                    self.origin_reward_idx = self.buffer_index
                    self.offset = -1*reward
                    self.og_reward[self.buffer_index] = 0.0
                    self.reward_sum = th.pow(self.og_reward + self.per_epsilon, self.per_alpha)
                    # calculate new max
                    self.max_reward_idx = th.argmax(self.reward_sum)
                    self.max_reward_sum = self.reward_sum[self.max_reward_idx]
                    self.pvalues = deepcopy(self.reward_sum)
                    self.pvalues[(self.e_sampled == 0).nonzero()] = self.max_reward_sum

                else:
                    self.og_reward[self.buffer_index] = self.offset + reward
                    
                    if self.buffer_index == self.origin_reward_idx:  # update offset if the current offset is overwritten
                        self.og_reward = self.og_reward - self.offset
                        self.origin_reward_idx = th.argmin(self.og_reward)
                        self.offset = -1*self.og_reward[self.origin_reward_idx].item()
                        self.og_reward = self.og_reward + self.offset
                        
                        self.reward_sum = th.pow(self.og_reward + self.per_epsilon, self.per_alpha)
                        self.max_reward_idx = th.argmax(self.reward_sum)
                        self.max_reward_sum = self.reward_sum[self.max_reward_idx]
                        self.pvalues = deepcopy(self.reward_sum)
                        self.pvalues[(self.e_sampled == 0).nonzero()] = self.max_reward_sum
                    
                    else:
                        self.reward_sum[self.buffer_index] = (self.og_reward[self.buffer_index] + self.per_epsilon)**self.per_alpha
                
            else:
                assert reward >= 0, "reward must be positive"
                self.reward_sum[self.buffer_index] = (reward + self.per_epsilon)**self.per_alpha
            
            if self.buffer_index == self.max_reward_idx:  # update max reward if current is overwritten
                self.max_reward_idx = th.argmax(self.reward_sum)
                self.max_reward_sum = self.reward_sum[self.max_reward_idx]
                # do we then scale the max p values... 
                
            if self.reward_sum[self.buffer_index] > self.max_reward_sum:
                self.max_reward_sum = self.reward_sum[self.buffer_index]
                self.max_reward_idx = self.buffer_index
            
            self.pvalues[self.buffer_index] = self.max_reward_sum
            self.e_sampled[self.buffer_index] = 0
            self.update(ep_batch.data.transition_data,
                        slice(self.buffer_index, self.buffer_index + ep_batch.batch_size),
                        slice(0, ep_batch.max_seq_length),
                        mark_filled=False)
            self.update(ep_batch.data.episode_data,
                        slice(self.buffer_index, self.buffer_index + ep_batch.batch_size))
            
            # record values for debugging/analysis
            self.reward_sum_record[self.buffer_counter] = (th.sum(ep_batch["reward"]) + self.per_epsilon)**self.per_alpha # NOTE needs adapting for offset
            self.sample_count[self.buffer_counter] = 0
            self.buffer_counter += ep_batch.batch_size
            
            # increment buffer index
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
        return self.episodes_in_buffer > batch_size


    def sample(self, batch_size, t):
        """Returns a sample of episodes from the replay buffer

        Args:
            batch_size (int): Number of episodes to return
            t (int): training timestep at which sampling is occuring, used to anneal per_beta
        """
        assert self.can_sample(batch_size)
        if self.episodes_in_buffer == batch_size:
            return self[:batch_size]
        
        else:
            probs = self.pvalues[:self.episodes_in_buffer]/th.sum(self.pvalues[:self.episodes_in_buffer], dim=0)  # calculate probability values
            ep_ids = np.random.choice(self.episodes_in_buffer, batch_size, replace=False, p=th.flatten(probs).cpu().detach().numpy())
            
            # Calculate importance sampling weights -- correct for bias introduced
            self.per_beta = self.per_beta_schedule.eval(t)
            is_weights = th.ones(batch_size, 1, 1) * 1/probs[ep_ids] * 1/self.episodes_in_buffer
            is_weights = th.pow(is_weights, self.per_beta)
            is_weights = is_weights/th.max(is_weights)  # normalise            
            self.data.transition_data["weights"][ep_ids]= is_weights
            
            # Update PER values for episodes sampled for first time # NOTE could be made more torchy
            for i in ep_ids:
                if not self.e_sampled[i]:
                    self.pvalues[i] = self.reward_sum[i]
                    self.e_sampled[i] = 1
                self.sample_count[i+(self.buffer_counter-self.episodes_in_buffer)] += 1
            return self[ep_ids]        

    def __repr__(self):
        return "PER ReplayBuffer. {}/{} episodes. Keys:{} Groups:{}".format(self.episodes_in_buffer,
                                                                        self.buffer_size,
                                                                        self.scheme.keys(),
                                                                        self.groups.keys())


def save_per_distributions(per_buffer, path):
    """ Saves PER distributions within the directory specified by `path`. 
    Path should not specify the file name.
    """
    print(f'saving PER objects to {path}')
    pvalues = th.flatten(per_buffer.pvalues).cpu().detach().numpy()
    reward_sum = th.flatten(per_buffer.reward_sum).cpu().detach().numpy()
    reward_sum_record = deepcopy(per_buffer.reward_sum_record)
    e_sampled = deepcopy(per_buffer.sample_count)
    per_beta = deepcopy(per_buffer.per_beta)
    
    offset = deepcopy(per_buffer.offset)
    ori = deepcopy(per_buffer.origin_reward_idx)
    og_rewards = th.flatten(per_buffer.og_reward).cpu().detach().numpy()
    
    th.save({"pvalues": pvalues,
             "reward_sum": reward_sum, 
             "reward_sum_record": reward_sum_record, 
             "sample_count": e_sampled, 
             "per_beta": per_beta,
             "offset": offset,
             "ori": ori,
             "og_rewards": og_rewards}, 
            "{}/per_objs.th".format(path))
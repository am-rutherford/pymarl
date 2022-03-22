""" Episode runner refactored to work with petting zoo api
https://www.pettingzoo.ml/api
"""

#from envs import REGISTRY as env_REGISTRY
from shutil import ExecError
from src.envs import REGISTRY as env_REGISTRY
from functools import partial
#from components.episode_buffer import EpisodeBatch
from src.components.episode_buffer import EpisodeBatch
import numpy as np
from src.utils.zoo_utils import update_batch_pre, quadratic_makespan_reward


class AsyncEpisodeRunner:
    
    debug = False

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.episode_limit = self.env.episode_limit
        self.t = 0
        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac

    def get_env_info(self):
        info = self.env.get_env_info()
        info["n_actions"] = 5 # NOTE to add None action
        return info
    
    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    def run(self, test_mode=False):
        if self.debug: print('*** reset environment ***')
        self.reset()

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)  # NOTE not sure what this is

        obs, reward, done, info = self.env.last()
        k = 0
        while not terminated:  
            k += 1  
                    
            pre_transition_data = self.env.get_pretran_data()
            
            if self.debug: 
                print(f'-- step {k} \nState: {self.env.state()}, Agent: {self.env.agent_selection}, Time: {self.env.sim_time()}')
                print(f"Pre transition data: {pre_transition_data}")

            self.batch.update(pre_transition_data, ts=self.t)
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
            action = actions[0][self.env.agent_idx()].item()
            
            if action == 4:
                self.env.step(None)  # terminated action to update env correctly
            else:
                self.env.step(action)
                
            obs, reward, done, env_info = self.env.last()
            
            if done:
                if self.debug: print(f'{self.env.agent_selection} done!')
                if len(self.env.agents) == 1: terminated = True
   
            if self.debug: print(f'Actions: {actions}\nReward {reward}, Time {self.env.sim_time()}')
            
            episode_return += reward
            post_transition_data = {
                    "actions": actions, 
                    "reward": [(reward,)],
                    "terminated": [[(terminated),]],  # NOTE used to be: [(terminated != env_info.get("episode_limit", False),)] # env info here is info from step()
                }

            self.batch.update(post_transition_data, ts=self.t)
            
            self.t += 1
            if self.t == self.episode_limit:
                terminated = True
            
        pre_transition_data = self.env.get_pretran_data()
        self.batch.update(pre_transition_data, ts=self.t)
        '''last_data = {
            "state": [self.env.state()],
            "avail_actions": [obs["action_mask"]],
            "obs": [obs["observation"]]
        }
        print('last data', last_data)
        self.batch.update(last_data, ts=self.t) '''
        # Select actions in the last stored state
        actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        self.batch.update({"actions": actions}, ts=self.t)
        #print('last data', pre_transition_data, 'actions', actions)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)
        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):  #-- can't rectify log_stat 
            self._log(cur_returns, cur_stats, log_prefix)
        
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env
        
        if self.debug: raise Exception()
        
        return self.batch

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()

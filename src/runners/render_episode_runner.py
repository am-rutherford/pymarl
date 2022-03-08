""" Episode runner refactored to work with petting zoo api
https://www.pettingzoo.ml/api
"""

#from envs import REGISTRY as env_REGISTRY
from math import floor
from shutil import ExecError
from src.envs import REGISTRY as env_REGISTRY
from functools import partial
#from components.episode_buffer import EpisodeBatch
from src.components.episode_buffer import EpisodeBatch
import numpy as np
from src.utils.zoo_utils import update_batch_pre, quadratic_makespan_reward
#from alex_4yp.camas_sim_vis import animate_des
from camas_gym.envs.rendering.zoo_camas_sim_vis import animate_des
import matplotlib.pyplot as plt

class RenderEpisodeRunner:
    
    debug = True

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1
        
        self.logger.console_logger.debug('Rendering runner initialised')

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
        test_mode = True 
        print('*** reset environment ***')
        self.reset()

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)  # NOTE not sure what this is

        obs, reward, done, info = self.env.last()
        last_time = self.env.sim_time()
        k = 0
        all_done = False
        while not terminated:  
            k += 1          
            if self.debug: print(f'-- step {k} \nState: {self.env.state()}, Agent: {self.env.agent_selection}, Time: {last_time}')
            
            
            pre_transition_data = self.env.get_pretran_data()
            if self.debug: print(f"Pre transition data: {pre_transition_data}")
            self.batch.update(pre_transition_data, ts=self.t)
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
            action = actions[0][self.env.agent_idx()].item()
            
            if action == 4:
                self.env.step(None)  # terminated action to update env correctly
            else:
                self.env.step(action)
                
            obs, _, done, env_info = self.env.last()
            reward = -1*(self.env.sim_time() - last_time)
            #reward = 0
            if done: reward += 20
            last_time = self.env.sim_time()
            
            if done and len(self.env.agents) == 1:
                all_done = True 
                terminated = True
                reward += quadratic_makespan_reward(last_time)
            
            reward = reward/100 # NOTE scaled down
            if self.debug: print(f'Actions: {actions}\nReward {reward}, Time {last_time}')
            episode_return += reward
            post_transition_data = {
                    "actions": actions, 
                    "reward": [(reward,)],
                    "terminated": [[(all_done),]],  # NOTE used to be: [(terminated != env_info.get("episode_limit", False),)] # env info here is info from step()
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
        
        #rc = {'agent_0':'blue', 'agent_1':'green', 'agent_2':'yellow'}
        rc = {'agent_'+str(i): self._get_agent_colour(i) for i in range(self.env.agent_count())}
        rs = {agent:'square' for agent in rc.keys()}

        print('events', self.env._events)
        print('len', len(self.env._events))
        aevents = {agent: [('node', self.env._tm.nodes[self.env.inital_state(agent)], 0.0)] for agent in rc.keys()}

        for event in self.env._events:
            for a in aevents.keys():
                if event[2] == a:
                    if event[0] == 'location':
                        if event[3] != "Agent reached goal":
                            aevents[a].append(('node', self.env._tm.nodes[event[3]], event[1]))
                    else:
                        aevents[a].append(event)
                continue 
        print('aevents', aevents)

        animate_des(self.env._tm, aevents, rc, rs) # current starts from teh second node visited

        #env._tm.draw()
        plt.show()
        
        raise Exception()

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

    def _get_agent_colour(self, idx):
        self._colours = ["blue", "green", "red", "cyan", "magenta", "yellow", "black"]
        
        if idx > len(self._colours):
            idx = idx - floor(idx % len(self._colours)) * self._colours
        return self._colours[idx]
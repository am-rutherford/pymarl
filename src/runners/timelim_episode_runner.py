
from runners import AsyncEpisodeRunner
from rapport_topological.navigation import construct_shortest_path_policy
from rapport_models.markov.state import State
import torch as th
import random 

class TimeLimEpisodeRunner(AsyncEpisodeRunner):
    
    debug = False
    
    def __init__(self, args, logger):
        super().__init__(args, logger)
        assert args.env == "camas", "Only Camas supported by this Runner"
        
        self.time_lim = self.env.map_param("time-lim")
        assert self.time_lim > 0, "time limit must be positive"
        self.logger.console_logger.info(f'Running Camas time limited episodes with a limit of {self.time_lim}')
        
        self.env.reset()
        self.sss_policies = {agent: construct_shortest_path_policy(self.env._tm, self.env._goal_states[agent]) 
            for agent in self.env.agents}
        
    
    def run(self, test_mode=False):
        if self.debug: print('*** reset environment ***')
        self.reset()

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)  # NOTE not sure what this is

        obs, reward, done, info = self.env.last()
        k = 0
        sss_actions = False
        while (not terminated):  
            k += 1  
                    
            pre_transition_data = self.env.get_pretran_data()
            
            if self.debug: 
                print(f'-- step {k} \nState: {self.env.state()}, Agent: {self.env.agent_selection}, Time: {self.env.sim_time()}')
                print(f"Pre transition data: {pre_transition_data}")

            self.batch.update(pre_transition_data, ts=self.t)
            if sss_actions:
                actions = self.select_sss_actions()
            else:
                actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
            action = actions[0][self.env.agent_idx()].item()
            
            if action == 4:
                self.env.step(None)  # terminated action to update env correctly
            else:
                self.env.step(action)
                
            obs, reward, done, env_info = self.env.last()
            
            if done:
                if self.debug: print(f'{self.env.agent_selection} done!')
                if len(self.env.agents) == 1: 
                    terminated = True
                    if self.debug: print(f'All agents done, camas time: {self.env.sim_time()}')
   
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
                print('episode lim reached')
                terminated = True
            if (self.env.sim_time() > self.time_lim) and not sss_actions: #self.time_lim:
                if self.debug: print(f'camas time: {self.env.sim_time()} greater than limit {self.time_lim}, switching to sss actions')
                sss_actions = True
            
        pre_transition_data = self.env.get_pretran_data()
        self.batch.update(pre_transition_data, ts=self.t)

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
    
    
    def select_sss_actions(self):
        acts = th.ones(1, self.args.n_agents, dtype=int)*4
        
        # choose action for agent acting
        agent = self.env.agent_selection
        agent_loc = self.env._agent_location[agent]
        agent_idx = self.env.agent_name_mapping[self.env.agent_selection]
        if self.debug:  print(f'choosing action for {agent}, loc: {agent_loc}, idx: {agent_idx}')
        
        # (random.uniform(0, 1) > self.epsilon) or
        if True:  # exploit
            camas_act = self.sss_policies[agent]._state_action_map[State({'loc': agent_loc})]
            if self.debug: print(f'sss action, camas act {camas_act}')
            
            if camas_act is None:
                action = 4
            else:
                action = self.env.to_gym_action(agent_loc, camas_act)

        else:  # explore
            avail_actions = self.batch["avail_actions"][:, self.t]
            
            action = random.choice([i for i, x in enumerate(avail_actions[0, agent_idx]) if x==1])
            if self.debug: print(f'random, action {action}, avail agent acts {avail_actions[0, agent_idx]}')
            
        acts[0, agent_idx] = action
        if self.debug: print(f'acts {acts}')
        return acts
        
    
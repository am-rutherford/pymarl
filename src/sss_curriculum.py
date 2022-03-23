""" Includes two functions which use shortest path policies
1) run_sss_curriculum - trains a PyMARL agent using experiences gathered 
    while following an epsilon greedy shortest path policy.

2) mean_sss_time - returns the mean time taken to complete a map while following
    an epsilon greedy shortest path policy.
"""

import datetime
import os
import time
from sympy import EX
import yaml
import torch as th
from types import SimpleNamespace as SN
from utils.logging import Logger
import numpy as np
import random
from logging import getLogger
from rapport_topological.navigation import construct_shortest_path_policy
from rapport_models.markov.state import State
from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot
from utils.logging import Logger

from src.components.episode_buffer import EpisodeBatch
from runners import AsyncEpisodeRunner
from main import recursive_dict_update
from run import args_sanity_check

def load_configs():
    """ Load configuration dictionaries from default locations
    """
    # Get the defaults from default.yaml
    with open(os.path.join(os.path.dirname(__file__), "config", "default.yaml"), "r") as f:
        try:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)
            
    # Get qmix params from qmix.yaml
    with open(os.path.join(os.path.dirname(__file__), "config", "algs", "qmix.yaml"), "r") as f:
        try:
            alg_dict = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)
            
    # Get camas params from camas.yaml
    with open(os.path.join(os.path.dirname(__file__), "config", "envs", "camas.yaml"), "r") as f:
        try:
            env_dict = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)
            
    config_dict = recursive_dict_update(config_dict, alg_dict)
    config_dict = recursive_dict_update(config_dict, env_dict)
    return config_dict

class SSS_Runner(AsyncEpisodeRunner):
    """ PyMARL Episode Runner for gathering shortest path based experience episodes
    """
    
    debug = False
    
    
    def __init__(self, args, logger, epsilon=0.3):
        super().__init__(args, logger)
        
        self.epsilon = epsilon
        
        self.env.reset()
        self.policies = {agent: construct_shortest_path_policy(self.env._tm, self.env._goal_states[agent]) 
            for agent in self.env.agents}


    def run(self) -> EpisodeBatch:
        """ Returns an transistions for one episode for an agent acting in an 
        epsilon greedy fashion while following its shortest path.
        """
        if self.debug: print('*** reset environment ***')
        self.reset()
        
        terminated = False
        episode_return = 0
        #self.mac.init_hidden(batch_size=self.batch_size)  # NOTE not sure what this is

        obs, reward, done, info = self.env.last()
        k = 0
        while not terminated:  
            k += 1  
                    
            pre_transition_data = self.env.get_pretran_data()
            
            if self.debug:
                print(f'-- step {k} \nState: {self.env.state()}, Agent: {self.env.agent_selection}, Time: {self.env.sim_time()}')
                print(f"Pre transition data: {pre_transition_data}")

            self.batch.update(pre_transition_data, ts=self.t)
            #actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=False)
            #print(f'actions {actions}, type {type(actions)}, size {actions.size()}')
            #print('my selector: ', self.select_actions())
            actions = self.select_actions()
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
        
        actions = self.select_actions()
        self.batch.update({"actions": actions}, ts=self.t)
        
        self.t_env += self.t
        
        return self.batch
        
    
    def select_actions(self) -> th.Tensor:
        """ Choose the action to stay on the shorest path or a random action 
        depending on epsilon test.
        """
        acts = th.ones(1, self.args.n_agents, dtype=int)*4
        
        # choose action for agent acting
        agent = self.env.agent_selection
        agent_loc = self.env._agent_location[agent]
        agent_idx = self.env.agent_name_mapping[self.env.agent_selection]
        if self.debug:  print(f'choosing action for {agent}, loc: {agent_loc}, idx: {agent_idx}')
        
        if random.uniform(0, 1) > self.epsilon:  # exploit
            camas_act = self.policies[agent]._state_action_map[State({'loc': agent_loc})]
            if self.debug: print(f'exploiting, camas act {camas_act}')
            
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

def run_sss_curriculum(args, logger,  num_episodes, train_steps, test_episodes):
    """Trains a PyMARL method using shortest path experiences and saves the result 
    to the results/model directory

    Args:
        num_episodes (int): number of experience episodes to gather
        train_steps (int): number of steps to train the model for
        test_episodes (int): number of episodes to evaluate the model on once training is complete
    """
    print(' -- Env args', args.env_args)
    start_time = time.time()
    
    main_runner = r_REGISTRY[args.runner](args=args, logger=logger)    
    sss_runner = SSS_Runner(args, logger)
    
    # Set up schemes and groups
    env_info = sss_runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]

    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {
        "agents": args.n_agents
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }
    
    buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                        preprocess=preprocess,
                        device="cpu" if args.buffer_cpu_only else args.device)

    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    # Give runners the scheme
    main_runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)
    sss_runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # Learner
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)
    if args.use_cuda:
        learner.cuda()
    
    logger.console_logger.info(f'...gathering data...')
    for _ in range(num_episodes):
        
        episode_batch = sss_runner.run()
        buffer.insert_episode_batch(episode_batch)
    
    logger.console_logger.info(f'...training network...')
    for i in range(train_steps):
        episode_sample = buffer.sample(args.batch_size)

        # Truncate batch to only filled timesteps
        max_ep_t = episode_sample.max_t_filled()
        episode_sample = episode_sample[:, :max_ep_t]

        if episode_sample.device != args.device:
            episode_sample.to(args.device)

        learner.train(episode_sample, sss_runner.t_env, i)
    
    tdelta = time.time()-start_time
    logger.console_logger.info(f'...time taken for training: {datetime.timedelta(seconds=tdelta)}...')
    
    # Evaluate trained agent
    logger.console_logger.info(f'...evaluating...')
    tt, sc, gc = [], [], []
    for _ in range(test_episodes):
        main_runner.run(test_mode=True)
        tt.append(main_runner.env.sim_time())
        sc.append(main_runner.env.step_count())
        gc.append(main_runner.env.agents_at_goal())
        
    logger.console_logger.info(f'-- evaluation av test time: {np.mean(tt)} ({np.var(tt)}), av step count {np.mean(sc)} ({np.var(sc)}), percentage at goal {np.mean(gc)} ({np.var(gc)}), {len(sc)} episodes')
    
    logger.console_logger.info('...saving  model...')
    save_path = os.path.join(args.local_results_path, "models", args.unique_token, str(main_runner.t_env))
    #"results/models/{}".format(unique_token)
    os.makedirs(save_path, exist_ok=True)
    logger.console_logger.info("Saving models to {}".format(save_path))

    # learner should handle saving/loading -- delegate actor save/load to mac,
    # use appropriate filenames to do critics, optimizer states
    learner.save_models(save_path) 
    
    # Save config
    with open(os.path.join(save_path, "config.yaml"), 'w') as outp:  # NOTE this has not been tested
        yaml.dump(args, outp)
    
    
def mean_sss_time(args, logger,  num_episodes, epsilon):
    """Runs a PyMARL-Camas map using an epsilon greedy shortest path policy

    Args:
        num_episodes (int): number of episodes to run for
        epislon (int): epsilon to use in action selection
    """
    print(' -- Env args', args.env_args)
    start_time = time.time()
    
    sss_runner = SSS_Runner(args, logger, epsilon=epsilon)
    
    # Set up schemes and groups
    env_info = sss_runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]

    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {
        "agents": args.n_agents
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }
    
    buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                        preprocess=preprocess,
                        device="cpu" if args.buffer_cpu_only else args.device)

    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    # Give runners the scheme
    #main_runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)
    sss_runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # Learner
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)
    if args.use_cuda:
        learner.cuda()
    
    logger.console_logger.info(f'...running {num_episodes} episodes...')
    episode_times = []
    step_count = []
    for i in range(num_episodes):
        _ = sss_runner.run()
        episode_times.append(sss_runner.env.sim_time())
        step_count.append(sss_runner.t)
        if i % 50 == 0:
            logger.console_logger.info(f'...{i} episodes complete...')
        
    print(f'Mean sim time for {num_episodes} and an epsilon of {epsilon}: {np.mean(episode_times)} ({np.var(episode_times)}), \
        mean step count {np.mean(step_count)} ({np.var(step_count)})')    
    return np.mean(episode_times), np.mean(step_count)
    

def load_default_params(map_name="bruno"):
    pass
    #TODO

    
if __name__ == "__main__":

    num_episodes = 10000
    train_steps = 40000
    test_episodes = 40
    
    console_logger = getLogger()
    logger = Logger(console_logger)
    
    config_dict = load_configs()  # NOTE should sanity check
    args = SN(**config_dict)  # gives attribute access to namespace
    args.use_cuda = False
    args.device = "cuda" if args.use_cuda else "cpu"
    
    unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    args.unique_token = unique_token
        
    mean_sss_time(args, logger, 100, 0)
    
    #run_sss_curriculum(args, logger, num_episodes, train_steps, test_episodes)
    



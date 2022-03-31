""" Includes two functions which use shortest path policies
1) run_sss_curriculum - trains a PyMARL agent using experiences gathered 
    while following an epsilon greedy shortest path policy.

2) mean_sss_time - returns the mean time taken to complete a map while following
    an epsilon greedy shortest path policy.
"""

import datetime
import os
from os.path import dirname, abspath
import time
from sympy import EX
import yaml
import torch as th
from types import SimpleNamespace as SN
from utils.logging import Logger, log_mac_weights
import numpy as np
import random
from logging import getLogger, INFO
from rapport_topological.navigation import construct_shortest_path_policy
from rapport_models.markov.state import State
from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot

from src.components.episode_buffer import EpisodeBatch
from runners import AsyncEpisodeRunner
from main import recursive_dict_update
from run import args_sanity_check

from torch.utils.tensorboard import SummaryWriter

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
    
    def __init__(self, args, logger, epsilon_mean=0.15, epsilon_var=0.1):
        super().__init__(args, logger)
        
        self.epsilon_mean = epsilon_mean
        self.epsilon_var = epsilon_var
        self.epsilon = self._draw_epsilon()
        
        self.env.reset()
        self.policies = {agent: construct_shortest_path_policy(self.env._tm, self.env._goal_states[agent]) 
            for agent in self.env.agents}


    def run(self) -> EpisodeBatch:
        """ Returns an transistions for one episode for an agent acting in an 
        epsilon greedy fashion while following its shortest path.
        """
        if self.debug: print('*** reset environment ***')
        self.reset()
        self.epsilon = self._draw_epsilon()
        
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
    
    def _draw_epsilon(self):
        epsilon = np.random.normal(self.epsilon_mean, self.epsilon_var)
        if epsilon < 0: epsilon = 0
        return epsilon
    
    def episode_makespan(self):
        return self.env.sim_time()
    

def run_sss_curriculum(args,
                       logger,
                       num_episodes,
                       max_train_steps,
                       test_makespan_cutoff,
                       test_episodes=20,
                       epsilon_mean=0.25,
                       epsilon_var=0.15,
                       log_freq=10000,
                       agent_weight_log_freq=20000):
    """Trains a PyMARL method using shortest path experiences and saves the result 
    to the results/model directory

    Args:
        num_episodes (int): number of experience episodes to gather
        max_train_steps (int): number of steps to train the model for
        test_episodes (int): number of episodes to evaluate the model on once training is complete
    """
    
    def _test_env(_runner, _test_episdoes):
        """ Test environment using `_runner`
        Returns:
            tt: test sim times
            sc: test step counts
            gc: test reached goal %'s
        """
        tt, sc, gc = [], [], []
        for _ in range(_test_episdoes):
            _runner.run(test_mode=True)
            tt.append(_runner.env.sim_time())
            sc.append(_runner.env.step_count())
            gc.append(_runner.env.agents_at_goal())
        return tt, sc, gc
    
    print(' -- Env args', args.env_args)
    start_time = time.time()
    
    tb_logs_direc = os.path.join(dirname(dirname(abspath(__file__))), "results", "curriculum_tb")
    tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(args.unique_token)
    logger.setup_tb(tb_exp_direc)
    
    args.log_interval = log_freq
    args.learner_log_interval = log_freq
    
    main_runner = r_REGISTRY[args.runner](args=args, logger=logger)    
    sss_runner = SSS_Runner(args, logger, epsilon_mean=epsilon_mean, epsilon_var=epsilon_var)
    
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
    
    ## --- Gather Data ---
    logger.console_logger.info(f'...gathering data...')
    ep_rewards = np.zeros(num_episodes)
    ep_epsilons = np.zeros(num_episodes)
    ep_times = np.zeros(num_episodes)
    ep_step_count = np.zeros(num_episodes)
    
    for k in range(num_episodes):    
        episode_batch = sss_runner.run()
        buffer.insert_episode_batch(episode_batch)
        ep_rewards[k] = th.sum(episode_batch["reward"])
        ep_epsilons[k] = sss_runner.epsilon
        ep_times[k] = sss_runner.episode_makespan()
        ep_step_count[k] = sss_runner.t
        if k % log_freq == 0:
            logger.console_logger.info(f'...{k} episodes complete, mean time {np.mean(ep_times)} ({np.std(ep_times)}), mean step count {np.mean(ep_step_count)} ({np.std(ep_step_count)})...')
            logger.console_logger.info(f'...mean rewards {np.mean(ep_rewards)} ({np.std(ep_rewards)}), mean epsilon {np.mean(ep_epsilons)} ({np.std(ep_epsilons)})')
    save_curriculum_data([ep_rewards, ep_epsilons, ep_times, ep_step_count])
    data_gathering_time = time.time() - start_time
    logger.console_logger.info(f'...time to gather {num_episodes} episodes: {datetime.timedelta(seconds=data_gathering_time)}, mean time {np.mean(ep_times)} ({np.std(ep_times)}), mean step count {np.mean(ep_step_count)} ({np.std(ep_step_count)})...')
    logger.console_logger.info(f'...mean rewards {np.mean(ep_rewards)} ({np.std(ep_rewards)}), mean epsilon {np.mean(ep_epsilons)} ({np.std(ep_epsilons)})')
    
    
    ## --- Train Network ---
    logger.console_logger.info(f'...training network...')
    for i in range(max_train_steps):
        episode_sample = buffer.sample(args.batch_size)

        # Truncate batch to only filled timesteps
        max_ep_t = episode_sample.max_t_filled()
        episode_sample = episode_sample[:, :max_ep_t]

        if episode_sample.device != args.device:
            episode_sample.to(args.device)

        learner.train(episode_sample, i, i)
        
        if i % log_freq == 0:
            tt, sc, gc = _test_env(main_runner, test_episodes)
            logger.log_stat("Test_mean_sim_time", np.mean(tt), i)
            logger.log_stat("Test_mean_step_count", np.mean(sc), i)
            logger.log_stat("Test_mean_goal_found", np.mean(gc), i)
            logger.console_logger.info(f'...logging at step {i}, mean sim time {np.mean(tt)}...')
            
            if np.mean(tt) < test_makespan_cutoff:
                tt, _, _ = _test_env(main_runner, test_episodes)
                if np.mean(tt) < test_makespan_cutoff:
                    logger.console_logger.info(f'Training passed evaluation at step {i}. Mean makespan: {np.mean(tt)}, cutoff: {test_makespan_cutoff}')
                    break
        
        if i % agent_weight_log_freq == 0:
            log_mac_weights(logger, mac, i)
    
    tdelta = time.time()-start_time
    logger.console_logger.info(f'...time taken for training: {datetime.timedelta(seconds=tdelta)}...')
    logger.console_logger.info(f'...time taken for data gathering: {datetime.timedelta(seconds=data_gathering_time)}...')
    
    ## --- Evaluate final agent ---
    logger.console_logger.info(f'...evaluating final agent...')
    tt, sc, gc = _test_env(main_runner, test_episodes)
    logger.log_stat("Test_mean_sim_time", np.mean(tt), i)
    logger.log_stat("Test_mean_step_count", np.mean(sc), i)
    logger.log_stat("Test_mean_goal_found", np.mean(gc), i)
    logger.console_logger.info(f'-- evaluation av test time: {np.mean(tt)} ({np.var(tt)}), av step count {np.mean(sc)} ({np.var(sc)}), percentage at goal {np.mean(gc)} ({np.var(gc)}), {len(sc)} episodes')
    
    logger.console_logger.info('...saving model...')
    save_path = os.path.join("curriculum", args.unique_token, str(main_runner.t_env))
    os.makedirs(save_path, exist_ok=True)
    logger.console_logger.info("Saving models to {}".format(save_path))

    # learner should handle saving/loading -- delegate actor save/load to mac,
    # use appropriate filenames to do critics, optimizer states
    learner.save_models(save_path) 
    
    # Save config
    with open(os.path.join(save_path, "config.yaml"), 'w') as outp:  # NOTE this has not been tested
        yaml.dump(args, outp)
        

def save_curriculum_data(array_to_save):
    save_path = os.path.join(args.local_results_path, "curriculum", "ep_data", args.unique_token)
    os.makedirs(save_path, exist_ok=True)
    np.save('{}/ep_data.npy'.format(save_path), array_to_save, allow_pickle=True)
    
    
def mean_sss_time(args, logger,  num_episodes, epsilon_mean):
    """Runs a PyMARL-Camas map using an epsilon greedy shortest path policy

    Args:
        num_episodes (int): number of episodes to run for
        epislon (int): epsilon to use in action selection
    """
    print(' -- Env args', args.env_args)
    start_time = time.time()
    
    sss_runner = SSS_Runner(args, logger, epsilon_mean=epsilon_mean, epsilon_var=0.0)
    
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
    rewards = []
    for i in range(num_episodes):
        batch = sss_runner.run()
        episode_times.append(sss_runner.env.sim_time())
        step_count.append(sss_runner.t)
        rewards.append(th.sum(batch["reward"]))
        if i % 50 == 0:
            logger.console_logger.info(f'...{i} episodes complete...')
        
    print(f'Mean sim time for {num_episodes} on {args.env_args["map_name"]} and an epsilon of {epsilon_mean}: {np.mean(episode_times)} ({np.var(episode_times)})')
    print(f'mean step count {np.mean(step_count)} ({np.var(step_count)}), mean reward: {np.mean(rewards)} ({np.var(rewards)})')    
    return np.mean(episode_times), np.mean(step_count)
    

def load_default_params(map_name="bruno"):
    pass
    #TODO

    
if __name__ == "__main__":

    ## *** Curriculum specific variables ***
    num_episodes = 50000
    train_steps_max = 300000
    test_episodes = 20
    test_makespan_cutoff = 50
    
    console_logger = getLogger()
    logger = Logger(console_logger)
    
    config_dict = load_configs()  # NOTE should sanity check
    args = SN(**config_dict)  # gives attribute access to namespace
    args.use_cuda = False
    args.device = "cuda" if args.use_cuda else "cpu"
    
    args.unique_token = "curriculum_{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    
    #args.batch_size = 64
    logger.console_logger.setLevel(INFO)
    
    #mean_sss_time(args, logger, 200, 0.0)
    if num_episodes > args.buffer_size:
        args.buffer_size = num_episodes
        print(f'Buffer size now {args.buffer_size}')
        
    run_sss_curriculum(args, logger, num_episodes, train_steps_max, test_makespan_cutoff,
                       test_episodes=test_episodes, log_freq=10000, agent_weight_log_freq=20000)
    



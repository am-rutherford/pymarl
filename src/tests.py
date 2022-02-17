import numpy as np
import torch as th
import os
import yaml
import time
from main import config_copy
from run import run, args_sanity_check
from utils.logging import get_logger
from types import SimpleNamespace as SN
from src.utils.timehelper import time_left, time_str

from main import recursive_dict_update
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from src.learners import REGISTRY as le_REGISTRY
from components.transforms import OneHot
from components.episode_buffer import ReplayBuffer

def load_configs():
    # Get the defaults from default.yaml
    with open(os.path.join(os.path.dirname(__file__), "config", "default.yaml"), "r") as f:
        try:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)
            
    # Get the defaults from default.yaml
    with open(os.path.join(os.path.dirname(__file__), "config", "algs", "qmix.yaml"), "r") as f:
        try:
            alg_dict = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)
            
    config_dict = recursive_dict_update(config_dict, alg_dict)
    return config_dict

if __name__ == '__main__':
    
    logger = get_logger()  # THIS causes issues
    
    config = load_configs()
    
    print('config', config)
    config['env'] = 'camas'
    #config["episode_limit"] = 250  # Use in creation of buffer - how many transistions can be stored
    config["runner"] = "async"
    config["mac"] = "basic_mac" #"zoo_mac"
    
    _config = args_sanity_check(config, logger)

    args = SN(**_config)  # gives attribute access to namespace
    args.device = "cuda" if args.use_cuda else "cpu"

    print('config', config)
    
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    print('env info', env_info)
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    
    print('obs shape **', env_info["obs_shape"] )
    # Default/Base scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"][0], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {
        "agents": args.n_agents  # NOTE how is this used? not sure
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }
    
    buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)

    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # Learner
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)
    
    # Traning variables
    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0
    
    start_time = time.time()
    last_time = start_time
    
    while runner.t_env <= args.t_max:
        
        episode_batch = runner.run()
        
        print('episode complete')
        
        buffer.insert_episode_batch(episode_batch)

        if buffer.can_sample(args.batch_size):
            print('sampling from buffer')
            episode_sample = buffer.sample(args.batch_size)

            # Truncate batch to only filled timesteps
            max_ep_t = episode_sample.max_t_filled()
            episode_sample = episode_sample[:, :max_ep_t]

            if episode_sample.device != args.device:
                episode_sample.to(args.device)

            learner.train(episode_sample, runner.t_env, episode)

        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        print('n_test runs', n_test_runs)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:

            #logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
            #logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
            #    time_left(last_time, last_test_T, runner.t_env, args.t_max), time_str(time.time() - start_time)))
            last_time = time.time()
            print('testing')
            last_test_T = runner.t_env
            for _ in range(n_test_runs):
                runner.run(test_mode=True)

        if args.save_model and (runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
            model_save_time = runner.t_env
            save_path = os.path.join(args.local_results_path, "models", args.unique_token, str(runner.t_env))
            #"results/models/{}".format(unique_token)
            os.makedirs(save_path, exist_ok=True)
            #logger.console_logger.info("Saving models to {}".format(save_path))

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            learner.save_models(save_path)

        episode += args.batch_size_run

        if (runner.t_env - last_log_T) >= args.log_interval:
            #logger.log_stat("episode", episode, runner.t_env)
            #logger.print_recent_stats()
            last_log_T = runner.t_env

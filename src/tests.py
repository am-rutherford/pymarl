import numpy as np
import torch as th
import os
import yaml
from main import config_copy
from run import run, args_sanity_check
from utils.logging import get_logger
from types import SimpleNamespace as SN

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
    
    logger = get_logger()
    
    config = load_configs()
    config['env'] = 'camas'
    config["episode_limit"] = 100  # Use in creation of buffer
    config["runner"] = "async"
    config["mac"] = "zoo_mac"
    
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
    
    runner.run()

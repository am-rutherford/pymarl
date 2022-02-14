import numpy as np
import torch as th
import os
import yaml
from main import config_copy
from run import run, args_sanity_check
from utils.logging import get_logger
from types import SimpleNamespace as SN
from runners import REGISTRY as r_REGISTRY




if __name__ == '__main__':
    
    logger = get_logger()
    
    # Get the defaults from default.yaml
    with open(os.path.join(os.path.dirname(__file__), "config", "default.yaml"), "r") as f:
        try:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)
            
    config = config_copy(config_dict)  
    config['env'] = 'sc2'
    #config["episode_limit"] = 100
    
    
    _config = args_sanity_check(config, logger)

    args = SN(**_config)  # gives attribute access to namespace
    args.device = "cuda" if args.use_cuda else "cpu"

    if args.runner != "episode": raise Exception('Incorrect runner')
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    
    print(args.n_agents, args.n_actions, args.state_shape)
    print(env_info)


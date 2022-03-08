from copy import deepcopy
import os
from sympy import EX
import yaml
import json
import sys
import collections
from pathlib import Path
from types import SimpleNamespace as SN
from logging import getLogger
import torch as th

from run import args_sanity_check, evaluate_sequential
from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot
from utils.logging import Logger

def run(args, _log):
        
    assert args.checkpoint_path != ""
    assert args.runner == "render"

    # Setup logging
    logger = Logger(_log)
    logger.console_logger.setLevel('DEBUG')

    runner = r_REGISTRY[args.runner](args=args, logger=logger)
    
    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]

    # Default/Base scheme
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
    logger.console_logger.debug(f"Buffer scheme: {scheme}, groups: {groups}")

    buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)

    
    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # Learner
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)


    timesteps = []
    timestep_to_load = 0

    if not os.path.isdir(args.checkpoint_path):
        logger.console_logger.info("Checkpoint directiory {} doesn't exist".format(args.checkpoint_path))
        return

    # Go through all files in args.checkpoint_path
    for name in os.listdir(args.checkpoint_path):
        full_name = os.path.join(args.checkpoint_path, name)
        # Check if they are dirs the names of which are numbers
        if os.path.isdir(full_name) and name.isdigit():
            timesteps.append(int(name))

    if args.load_step == 0:
        # choose the max timestep
        print('timesteps', timesteps)
        timestep_to_load = max(timesteps)
    else:
        # choose the timestep closest to load_step
        timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

    print(f'loading model timestep {timestep_to_load}')
    model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

    logger.console_logger.info("Loading model from {}".format(model_path))
    learner.load_models(model_path)
    runner.t_env = timestep_to_load

    tt, sc, gc = [], [], []
    for _ in range(30): # NOTE hard coded
        runner.run(test_mode=True)
        if args.env == "camas":
                    tt.append(runner.env.sim_time())
                    sc.append(runner.env.step_count())
                    gc.append(runner.env.agents_at_goal())


    '''if args.evaluate or args.save_replay:
        evaluate_sequential(args, runner)
        return'''
    
    
def _get_config(params, arg_name, subfolder):
    config_name = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del params[_i]
            break

    if config_name is not None:
        with open(os.path.join(os.path.dirname(__file__), "config", subfolder, "{}.yaml".format(config_name)), "r") as f:
            try:
                config_dict = yaml.load(f, yaml.FullLoader)
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)
        return config_dict
    

def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def _get_config_from_sacred(params): 
    dir = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == "--sacred":
            dir = _v.split("=")[1]
            del params[_i]
            break
    print('DIR', dir)
    if dir is not None:
        with open(os.path.join(Path(os.path.dirname(__file__)).parent, "results", "sacred", dir, "config.json"), "r") as f:
            #try:
            config_dict = json.load(f)#, yaml.FullLoader)
            '''except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)'''
        return config_dict

def _get_checkpoint_path(params):
    dir = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == "--model":
            dir = _v.split("=")[1]
            del params[_i]
            break
    
    if dir is not None:
        return os.path.join(Path(os.path.dirname(__file__)).parent.absolute(), "results", "models", dir)

if __name__ == "__main__":
    
    logger = getLogger()
    params = deepcopy(sys.argv)
    # Load config from sacred
    config_dict = _get_config_from_sacred(params)
    
    
    print('** config **', config_dict)
    
    #config_dict = args_sanity_check(config_dict, logger) 
    
    args = SN(**config_dict)  # gives attribute access to namespace
    
    # Set args for testing
    args.runner = "render"
    args.save_model = False
    args.use_tensorboard = False
    args.load_step = 0
    args.use_cuda = False
    args.device = "cuda" if args.use_cuda else "cpu"
    args.checkpoint_path = _get_checkpoint_path(params)
    
    print('args', args)
    print('check', args.checkpoint_path)
    
    run(args, logger)
    
    
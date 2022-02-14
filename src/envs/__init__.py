from functools import partial
from smac.env import MultiAgentEnv, StarCraft2Env
from camas_gym.envs.camas_zoo_masking import CamasZooEnv
from pettingzoo.butterfly.prison_v3 import env as PrisonEnv
import sys
import os

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
REGISTRY["camas"] = partial(env_fn, env=CamasZooEnv)
REGISTRY["prison"] = partial(env_fn, env=PrisonEnv)

if sys.platform == "linux":
    os.environ.setdefault("SC2PATH",
                          os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))

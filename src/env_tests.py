from pettingzoo.butterfly import prison_v3

from camas_gym.envs.camas_zoo_masking import CamasZooEnv

env = CamasZooEnv()

env.reset()
print('last', env.last())


print('state', env.state())

print(env.get_env_info()["obs_shape"].shape)

from pettingzoo.butterfly import prison_v3
import numpy as np

env = prison_v3.env(vector_observation=True)
env.reset()
print('state', np.shape(env.state()))
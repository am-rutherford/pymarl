from flatland.envs import rail_env    
from flatland.utils import flat
from flatland import 
env = flatland_env.env(environment=rail_env, use_renderer=True)
seed = 11
env.reset(random_seed=seed)
step = 0
ep_no = 0
frame_list = []
while ep_no < total_episodes:
    for agent in env.agent_iter():
        obs, reward, done, info = env.last()
        # act = env_generators.get_shortest_path_action(env.environment, get_agent_handle(agent))
        act = 2
        all_actions_pettingzoo_env.append(act)
        env.step(act)
        frame_list.append(PIL.Image.fromarray(env.render(mode='rgb_array')))
        step += 1
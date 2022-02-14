from sumo_rl import env as sumo_env


env = sumo_env(net_file='sumo_net_file.net.xml',
                route_file='sumo_route_file.rou.xml',
                use_gui=True,)
env.reset()

# 57922

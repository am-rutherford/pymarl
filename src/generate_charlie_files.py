import yaml
from sss_curriculum import load_configs
from types import SimpleNamespace as SN

from camas_gym.envs.camas_zoo_masking import CamasZooEnv

def main():

    config_dict = load_configs()  # NOTE should sanity check
    args = SN(**config_dict)  # gives attribute access to namespace
    
    
    env = CamasZooEnv(**args.env_args)
    
    tm = env._tm
    map_name = env.map_name
    #print(tm.nodes)
    map_config = {"bi": False, "nodes": []}
    print(tm.nodes["WayPoint2"].pose)
    for node in tm.nodes:
        nodeObj = tm.nodes[node]
        print(nodeObj.edges)
        print('pos', [nodeObj.pose.position.x, nodeObj.pose.position.y] + 6*[0])#  nodeObj.pose.orientation))
        nl = {"name": nodeObj.name, "pos": [nodeObj.pose.position.x, nodeObj.pose.position.y] + 6*[0]}
        raise Exception()
        map_config["nodes"]
        
    
    
    

if __name__ == "__main__":
    main()



import yaml
import os
import numpy as np
from sss_curriculum import load_configs
from types import SimpleNamespace as SN

from camas_gym.envs.camas_zoo_masking import CamasZooEnv
from camas.camas_utils import fit_ptd_from_mean_var
from rapport_topological import topological_map
from shapely.geometry import Point
from rapport_maths.geometry import Pose


def _generateMapFromCharlieData(map_data):
    """ Builds a rapport topological map """
    
    # Create topological nodes
    nodes = {node["name"]: topological_map.TopologicalNode(name=node["name"], pose=Pose(point=Point(node["pose"][:2])))
             for node in map_data["nodes"]}
    
    # Create edges
    edges = {}
    edge_to_id = {}
    for edge in map_data["edges"]:
        node1 = nodes[edge["start"]]
        node2 = nodes[edge["end"]]
        label = node1.name + '_' + node2.name
        edges[label] = node1.connect_to(node2, label, action=edge["action"])
        edge_to_id[label] = edge["id"]
        if map_data["bi"]:
            label = node2.name + '_' + node1.name
            edges[label] = node2.connect_to(node1, label, action=edge["action"])
            edge_to_id[label] = edge["id"]
    
    return topological_map.TopologicalMap(nodes=nodes,
                            edges=edges,
                            name='prebuilt'), edge_to_id
    

def create_bands_ptds(tm, mean=1, var=0.1, bands=4):
    """ Very simple congestion band and PTD creation
        4 congestion bands for each edge
        Same PTD for each congestion band on each edge
    
    Args:
        tm: Rapport topological map
    Returns:
        bands: Congestion bands dictionary (or context dictionary if 
               use_fine_grain_context is True)
        ptds: PTD dictionary
    """

    congestion_bands = []
    congestion_mean_var = []
    for i in range(bands):
        if i == (bands-1):
            congestion_bands.append((i, np.inf))
        else:
            congestion_bands.append((i, i))
        norm_mean, norm_var = i*mean+1, var*(i+1)  # Arbritary choices
        ln_mean = np.exp(norm_mean + (norm_var/2.0))
        ln_variance = ((np.exp(norm_var) - 1) * 
                        np.exp((2 * norm_mean) + norm_var))
        congestion_mean_var.append((ln_mean, ln_variance))
    bands = {edge:congestion_bands for edge in tm.edges}

    ptds = {edge:[ fit_ptd_from_mean_var(mean, var)
            for mean, var in congestion_mean_var ]
            for edge in tm.edges}

    return bands, ptds


def generateCharlieTMFile(name):
    """ For use on Alex's machine. 
    Generates a topolocial map file that can then be used to generated the 
    Camas ptd specification file """

    config_dict = load_configs()  
    args = SN(**config_dict)  # gives attribute access to namespace
    env = CamasZooEnv(**args.env_args)
    tm = env._tm
    
    #raise Exception()
    map_name = env.map_name
    #print(tm.nodes)
    map_config = {"bi": False, "nodes": [], "edges": [], "durations": []}
    for node in tm.nodes:
        nodeObj = tm.nodes[node]
        map_config["nodes"].append({"name": nodeObj.name,
                                    "pose": [float(nodeObj.pose.position.x), float(nodeObj.pose.position.y)] + 6*[0],
                                    })
        
        for edge in nodeObj.edges:
            eObj = nodeObj.edges[edge]
            if env._anna_map:
                eid = eObj.edge_id[2:]
            else:
                eid = eObj.edge_id
            el = {"id": eid,
                  "start": eObj.start.name,
                  "end": eObj.end.name,
                  "distance": 1,
                  "bands": [], # 1, 2
                  "action": "move_base"}
            
            map_config["edges"].append(el)
    
    # write
    print('..writing map config..')
    path = os.path.join(os.getcwd(), "{}.yaml".format(name))
    #os.makedirs(path, exist_ok=True)
    with open(path, 'w') as outp:
        yaml.dump(map_config, outp)
        

def addPTDs(map_file_path, ptd_dir, map_file_final_path=None, two_bands=False):
    """ For use on Charlie's machine. 
    
    For a specified topolgocial map, this function: 
        1. Loads the map config file and generates the topological map
        2. Generates ptds for each band
        3. Saves ptds to the `ptd_dir` directory
        4. Updates the map config file to have the ptds
        
    If map_file_final_path is set to None, the original config file will be updated
    with the ptds. If two bands is set to true, only two congestion bands will be created
    with the second having high congestion.
    """
    if map_file_final_path is None:
        map_file_final_path = map_file_path
        
    with open(map_file_path) as file:
        print(f'...reading in yaml: {map_file_path}...')
        map_config = yaml.load(file, Loader=yaml.FullLoader)
    
    tm, _ = _generateMapFromCharlieData(map_config)
    if two_bands:
        _, ptds = create_bands_ptds(tm, mean=2, bands=2)
    else:
        bands, ptds = create_bands_ptds(tm)
    
    for node in tm.nodes:
        for edge in tm.nodes[node].edges:
            eptds = ptds[edge]
            file_paths = []
            for i in range(len(eptds)):
                path = os.path.join(ptd_dir,'{}_{}.csv'.format(edge, i))
                eptds[i].write_distribution_to_file(path)
                file_paths.append(path)
            
            map_config["durations"].append({'edge':edge, 'files':file_paths})
            
    # write
    print('..writing updated map config..')
    #os.makedirs(path, exist_ok=True)
    with open(map_file_final_path, 'w') as outp:
        yaml.dump(map_config, outp)

if __name__ == "__main__":
    """ 
    So this is a little more convoluted than I think we thought as I can't transfer the ptd
    file objects to you easily, the easiest way (I think) is for you(/python) to generate them. Thus
    these functions accomplish this. 
    
    All that needs to be done is to run the `addPTDs` function which will update the map config
    to have the ptd distributions. The map config file path needs to be specified along with where
    the ptds should be saved. Example:
    
    map_path = "/Users/alexrutherford/repos/pymarl/bruno.yaml"
    ptds_dir = "/Users/alexrutherford/repos/pymarl/ptds/bruno/" 
    
    I would save the ptds for each map into a seperate directory as that way things won't get confused
    as many maps have edges with the same name.
    """
    name = "supermarket-medium"
    generateCharlieTMFile("{}".format(name))
    
    '''map_path = 
    ptds_dir = 
    addPTDs(map_path, ptds_dir)'''
    
    '''map_path = "/Users/alexrutherford/repos/pymarl/{}.yaml".format(name)
    ptds_dir = "/Users/alexrutherford/repos/pymarl/ptds/{}/".format(name) 
    os.makedirs(ptds_dir, exist_ok=True)
    map_path_new =  "/Users/alexrutherford/repos/pymarl/{}_new.yaml".format(name)
    addPTDs(map_path, ptds_dir, map_path_new, two_bands=True)'''



from camas_gym.envs.camas_zoo_masking import MOVES
import numpy as np

def batch_update_creation(buffer, env):
    
    agent_idx = env.agent_idx()
    
    obs, avail_acts = np.array([]), np.array([])  # NEED TO DO THIS WITH NUMPY
    for agent in env.possible_agents:
        observation = env.observe(agent)
        obs = np.append(obs, observation["observation"] )
        #obs.append(observation["observation"])  # WRONG NEEDS TO BE 2D -- or can reshape
        
        if agent == env.agent_selection:
            avail_acts = np.append(avail_acts, observation["action_mask"])
            avail_acts = np.append(avail_acts, np.array([0.0]))            
        else:
            one_hot = np.zeros(5)  # NOTE find a better way!
            if agent in env.agents:
                print('one hot', one_hot, 'idx', env.agent_action(agent))
                agent_act = env.agent_action(agent)
            else: # terminal agent
                agent_act = None
                
            if agent_act is None: agent_act = -1
            one_hot[agent_act] = 1

            print('one hot', one_hot)
            avail_acts = np.append(avail_acts, one_hot)

    print('obs', obs, 'size', buffer.scheme['obs'], 'avial', avail_acts, 'size', buffer.scheme['avail_actions'])
    obs.resize((3,3))
    avail_acts.resize((3, 5))
            
    pre_transition_data = {
                "state": [env.state()],
                "obs": [obs],
                "avail_actions": [avail_acts]
    }
    print('pre tran', pre_transition_data)
    return pre_transition_data
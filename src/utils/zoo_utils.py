from camas_gym.envs.camas_zoo_masking import MOVES, CamasZooEnv
import numpy as np
    

def update_batch_pre(env, done): # buffer may not be the correct terminology
    """Creates pre transition buffer data
    
    Only one agent may act a time, other agents either carry out their current action again
    or choose None (idx = num_actions) if they are at their goal 

    Args:
        buffer (_type_): _description_
        time (_type_): _description_
        env (_type_): _description_
    """
        
    obs, avail_acts = np.array([]), np.array([])  
    for agent in env.possible_agents:
        observation = env.observe(agent)
        obs = np.append(obs, observation["observation"] )
        
        if agent == env.agent_selection and done:
            one_hot = np.zeros(5)
            one_hot[-1] = 1
            avail_acts = np.append(avail_acts, one_hot)
            
        elif agent == env.agent_selection:
            avail_acts = np.append(avail_acts, observation["action_mask"])
            avail_acts = np.append(avail_acts, np.array([0.0])) 
                       
        else:
            one_hot = np.zeros(5)  # NOTE find a better way!
            if agent in env.agents:
                agent_act = env.agent_action(agent)
            else: # terminal agent
                agent_act = None
                
            if agent_act is None: agent_act = -1
            one_hot[agent_act] = 1

            avail_acts = np.append(avail_acts, one_hot)
    #print('- pre tran created')
    obs.resize((3,len(observation["observation"])))
    avail_acts.resize((3, 5))
    pre_transition_data = {
                "state": [env.state()],
                "obs": [obs],
                "avail_actions": [avail_acts]
    }
    
    return pre_transition_data


def quadratic_makespan_reward(x):
    
    return ((x-100)**2)/100
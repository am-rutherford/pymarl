from operator import mod
import numpy as np
import os



if __name__ == "__main__":
    
    dir_path = "results/curriculum/ep_data"
    model_id = "curriculum_qmix__2022-03-29_23-42-30"
    model_id = "curriculum_qmix__2022-03-29_23-48-15"
    model_id = "curriculum_qmix__2022-03-29_23-51-33"
    load_path = dir_path + "/" + model_id + "/ep_data.npy"

    data = np.load(load_path)
    
    print('data', data)   
    for a in data:
        print(f'mean {np.mean(a)} ({np.std(a)})') 
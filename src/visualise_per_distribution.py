from json import load
import yaml
import os
import logging
import torch as th
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path
path = Path(os.path.dirname(__file__))
print(path.absolute())


def _get_(params, arg_name, subfolder):
    config_name = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del params[_i]
            break

    if config_name is not None:
        with open(os.path.join(os.path.dirname(__file__), "models", subfolder, "{}.yaml".format(config_name)), "r") as f:
            try:
                config_dict = yaml.load(f, yaml.FullLoader)
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)
        return config_dict


def run(model_id):
    
    def _movingaverage(interval, window_size=100):
        window = np.ones(int(window_size))/float(window_size)
        return np.convolve(interval, window, 'same')
    
    buffer_size = 5000
    
    ## load data
    load_path = os.path.join(os.getcwd(), "results", "models", model_id)
    if not os.path.isdir(load_path):
        #logger.console_logger.info("Checkpoint directiory {} doesn't exist".format(args.checkpoint_path))
        print(f'{load_path} does not exist')
        return

    # Go through all files in model directory
    timesteps = []
    for name in os.listdir(load_path):
        full_name = os.path.join(load_path, name)
        # Check if they are dirs the names of which are numbers
        if os.path.isdir(full_name) and name.isdigit():
            timesteps.append(int(name))

    if True:
        # choose the max timestep
        print('timesteps', timesteps)
        timestep_to_load = max(timesteps)
    else:
        # choose the timestep closest to load_step
        timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

    model_path = os.path.join(load_path, str(timestep_to_load))
    print("Loading model from {}".format(model_path))
    
    per_data = th.load(os.path.join(model_path, "per_objs.th"))
    print('per_data keys', per_data.keys(), 'epsiodes:', len(per_data["reward_sum_record"]))
    
    ## Process data
    all_r_values = th.stack(list(per_data["reward_sum_record"].values())).cpu().detach().numpy().flatten()
    
    if len(per_data["reward_sum_record"]) > buffer_size:
        offset = len(per_data["reward_sum_record"]) % buffer_size
        
        pvalues = np.concatenate((per_data["pvalues"][offset:], per_data["pvalues"][:offset]))  # order pvalues to line up with reward_sum_record
        x_offset = len(per_data["reward_sum_record"]) - buffer_size
        px = np.arange(x_offset, x_offset + buffer_size)
        
        buffer_filled = buffer_size
    else:
        x_offset = 0
        pvalues = per_data["pvalues"][:len(all_r_values)]
        px = np.arange(x_offset, x_offset + buffer_size)
        
        buffer_filled = len(all_r_values)
        
        
    buffer_sample_values = np.array(list(per_data["sample_count"].values())[x_offset:buffer_filled])
    
    sorted_indicies = np.argsort(per_data["reward_sum"][:buffer_filled])
    
    '''ts = pd.Series(buffer_sample_values[sorted_indicies])
    #ts.plot(style='k--')
    ts.rolling(window=60).mean().plot(style='k')

    # add the 20 day rolling standard deviation:
    ts.rolling(window=20).std().plot(style='b')
    plt.show()'''
    
    fig, axes = plt.subplots(2, 2)
    print('axes', axes)
    
    ## Probability and reward sorted distributions
    axes[0][0].plot(per_data["reward_sum"][sorted_indicies])
    axes[0][0].legend(["Sorted reward sum"])
    axes[1][0].plot(per_data["pvalues"][sorted_indicies]/np.sum(per_data["pvalues"][sorted_indicies]))
    axes[1][0].legend(["Sorted probabilty values"])
    
    sampled_val_movavg = _movingaverage(buffer_sample_values[sorted_indicies], 5)
    axes[0][1].plot(buffer_sample_values[sorted_indicies])
    axes[0][1].plot(sampled_val_movavg)
    axes[0][1].legend(["Sorted sample count", "mean"])
    '''axes[0].plot(px, pvalues+2)
    axes[0].plot(r_values)'''
    #ax.plot(list(per_data["sample_count"].values()))
    #ax.legend(["pvals", "rsum", "rsum sorted"])'''
    
    
    '''axes[3].plot(per_data["reward_sum"])
    axes[3].plot(all_r_values, alpha=0.5)
    axes[3].legend(["all reward values", "current buffer"])
    
    #axes[2].set_title('Sample count')
    axes[2].plot(per_data["sample_count"].values())
    axes[2].plot([len(per_data["sample_count"])-buffer_size, len(per_data["sample_count"])], [0, 0])
    axes[2].legend(["sample count", "current buffer"])'''
    
    #print('difference', np.sum(np.abs(r_values-per_data["reward_sum"][:2633])))
    
    plt.show()
    


if __name__ == "__main__":
    model_id = "qmix__2022-03-21_10-28-46" # smac
    model_id = "qmix__2022-03-20_20-40-44" # super small 5 a
    model_id = "qmix__2022-03-20_23-07-49" # super med 5 a
    model_id = "qmix__2022-03-21_10-20-16" # super med 5 a shorter ep time
    model_id = "qmix__2022-03-22_12-29-40" # pls no broke
    model_id = "qmix__2022-03-22_12-47-45" # pls sample count now work
    model_id = "qmix__2022-03-22_15-48-01"
    model_id = "qmix__2022-03-22_16-38-40"
    model_id = "qmix__2022-03-22_16-49-33"
    model_id = "qmix__2022-03-22_17-39-55" # supermarket smol
    model_id = "qmix__2022-03-22_22-04-05" # time-cost
    model_id = "qmix__2022-03-22_22-54-53" # bug fix lol lol
    
    run(model_id)
    
    
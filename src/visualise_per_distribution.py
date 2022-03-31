from json import load
from time import time
import yaml
import os
import logging
import torch as th
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

def load_per_objs(model_id):
    ## load data
    load_path = os.path.join(os.getcwd(), "results", "models", model_id)
    if not os.path.isdir(load_path):
        #logger.console_logger.info("Checkpoint directiory {} doesn't exist".format(args.checkpoint_path))
        #print(f'{load_path} does not exist')
        raise Exception(f'{load_path} does not exist')

    # Go through all files in model directory
    all_timesteps = []
    for name in os.listdir(load_path):
        full_name = os.path.join(load_path, name)
        # Check if they are dirs the names of which are numbers
        if os.path.isdir(full_name) and name.isdigit():
            all_timesteps.append(int(name))
    all_timesteps.sort()
    timesteps = all_timesteps[1:]
    t_max = max(timesteps)
    per_data = {}
    for t in timesteps:
        model_path = os.path.join(load_path, str(t))
        per_data[t] = th.load(os.path.join(model_path, "per_objs.th"))
    return per_data, timesteps

def run_per(model_id):
    
    def _process_buffer(_per_data, _buffer_size):
        data = {}
        data["rsum_record"] = th.stack(list(_per_data["reward_sum_record"].values())).cpu().detach().numpy().flatten()
        data["ecount"] = len(data["rsum_record"])
        
        if data["ecount"] > _buffer_size:
            offset = len(_per_data["reward_sum_record"]) % _buffer_size
        
            data["pval_record"] = np.concatenate((_per_data["pvalues"][offset:], _per_data["pvalues"][:offset]))  # order pvalues to line up with reward_sum_record
            x_offset = len(_per_data["reward_sum_record"]) - _buffer_size
            data["pval_recordx"] = np.arange(x_offset, x_offset + _buffer_size)
            
            data["e_in_buffer"] = _buffer_size
        else:
            x_offset = 0
            data["pval_record"] = _per_data["pvalues"][:data["ecount"]]
            data["pval_recordx"] = np.arange(x_offset, x_offset + _buffer_size)
            
            data["e_in_buffer"] = data["ecount"]
            
        data["buffer_sample_count"] = np.array(list(_per_data["sample_count"].values())[x_offset:x_offset+data["e_in_buffer"]])
        data["buffer_sortedidx"] = np.argsort(_per_data["reward_sum"][:data["e_in_buffer"]])
        
        return data
    
    def _movingaverage(interval, window_size=100):
        window = np.ones(int(window_size))/float(window_size)
        return np.convolve(interval, window, 'same')
    
    buffer_size = 5000
    
    ## load data
    per_data, timesteps = load_per_objs(model_id)
    per_proccessed = {t: _process_buffer(per_data[t], buffer_size) for t in timesteps}
    t_max = max(timesteps)
    print('per_data loaded. Keys:', per_data[t_max].keys(), ' epsiodes:', len(per_data[t_max]["reward_sum_record"]))

    ## PLOTLY
    #fig = go.Figure()
    fig = make_subplots(2, 2,
                        subplot_titles=("Plot 1", "Plot 2", "Plot 3", "Plot 4"))
    
    # Add traces, one for each slider step     
    for t in timesteps:
        sorted_indicies = per_proccessed[t]["buffer_sortedidx"]
        fig.add_trace(
            go.Scatter(
                visible=False,
                line=dict(color="#00CED1", width=1),
                name="timestep: " + str(t),
                y=per_data[t]["reward_sum"][sorted_indicies],
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                visible=False,
                line=dict(color="#d12600", width=1),
                name="timestep: " + str(t),
                y=per_data[t]["pvalues"][sorted_indicies]/np.sum(per_data[t]["pvalues"][sorted_indicies]),
            ),
            row=2,
            col=1,
        )
        
        fig.add_trace(
            go.Scatter(
                visible=False,
                line=dict(color="#d12600", width=1),
                name="timestep: " + str(t),
                y=per_proccessed[t]["buffer_sample_count"][sorted_indicies],
            ),
            row=1,
            col=2,
        )
        
    # Make 1st trace visible
    fig.data[3].visible = True
    fig.data[4].visible = True
    fig.data[5].visible = True

    # Create and add slider
    steps = []
    for i in range(len(timesteps)):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)},
                {"title": "Slider switched to timestep: " + str(timesteps[i])}],  # layout attribute
        )
        step["args"][0]["visible"][3*i] = True  # Toggle i'th trace to "visible"
        step["args"][0]["visible"][3*i+1] = True
        step["args"][0]["visible"][3*i+2] = True
        steps.append(step)

    sliders = [dict(
        active=1,
        currentvalue={"prefix": "Timestep: "},
        pad={"t": 50},
        steps=steps
    )]

    fig.update_layout(
        sliders=sliders
    )
    rmax = max([max(per_data[t]["reward_sum"]) for t in timesteps])*1.1
    pmax = max([max(per_data[t]["pvalues"][per_proccessed[t]["buffer_sortedidx"]]/np.sum(per_data[t]["pvalues"][per_proccessed[t]["buffer_sortedidx"]])) for t in timesteps])*1.1
    print('r range', [0, (per_proccessed[t_max]["rsum_record"])])
    fig.update_yaxes(title_text="yaxis 1 title", range=[0, rmax], row=1, col=1)
    fig.update_yaxes(title_text="yaxis 2 title", range=[0, pmax], row=2, col=1)
    fig.update_yaxes(title_text="yaxis 3 title", range=[0, max(list(per_data[t_max]["sample_count"].values()))*1.1], row=1, col=2)
    #fig.update_yaxes(title_text="yaxis 4 title", row=2, col=2)

    fig.show()  
    
    fig2 = go.Figure()
    fig2.add_trace(
            go.Scatter(
                visible=True,
                line=dict(color="#00CED1", width=1),
                name="sample count",
                y=list(per_data[max(timesteps)]["sample_count"].values()),
            ),
            )
    fig2.show()

def run_regular(model_id):
    
    def _process_data(_data):
        pdata = {}
        pdata["rsum_record"] = th.stack(list(_data["reward_sum_record"].values())).cpu().detach().numpy().flatten()
        pdata["sample_count"] = np.array(list(_data["sample_count"].values()))
        return pdata
    
    buffer_data, timesteps = load_per_objs(model_id)
    processed_data = {t: _process_data(buffer_data[t]) for t in timesteps}
    
    print(f'data {buffer_data[max(timesteps)].keys()}')
    
    fig = make_subplots(2, 1,
                        subplot_titles=("Plot 1", "Plot 2"))
    
    # Add traces, one for each slider step     
    for t in timesteps:
        fig.add_trace(
            go.Scatter(
                visible=False,
                #line=dict(color="#00CED1", width=1),
                mode='markers',
                name="timestep: " + str(t),
                y=processed_data[t]["rsum_record"],
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                visible=False,
                #line=dict(color="#d12600", width=1),
                mode='markers',
                name="timestep: " + str(t),
                y=processed_data[t]["sample_count"],
            ),
            row=2,
            col=1,
        )
        
    # Make 1st trace visible
    fig.data[0].visible = True
    fig.data[1].visible = True

    # Create and add slider
    steps = []
    for i in range(len(timesteps)):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)},
                {"title": "Slider switched to timestep: " + str(timesteps[i])}],  # layout attribute
        )
        step["args"][0]["visible"][2*i] = True  # Toggle i'th trace to "visible"
        step["args"][0]["visible"][2*i+1] = True
        steps.append(step)

    sliders = [dict(
        active=1,
        currentvalue={"prefix": "Timestep: "},
        pad={"t": 50},
        steps=steps
    )]

    fig.update_layout(
        sliders=sliders
    )
    rmin = min(processed_data[max(timesteps)]["rsum_record"])
    rmax = max(processed_data[max(timesteps)]["rsum_record"])*1.1
    #rmax = max([max(processed_data[t]["rsum_record"]) for t in timesteps])*1.1
    fig.update_yaxes(title_text="yaxis 1 title", range=[rmin, rmax], row=1, col=1)
    fig.update_yaxes(title_text="yaxis 3 title", range=[0, max(list(processed_data[max(timesteps)]["sample_count"]))*1.1], row=2, col=1)
    #fig.update_yaxes(title_text="yaxis 4 title", row=2, col=2)

    fig.show()  

def old_code(per_data):
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
        
        
    buffer_sample_values = np.array(list(per_data["sample_count"].values())[x_offset:x_offset+buffer_filled])
    
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
    #print('s', buffer_sample_values)
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
    model_id = "qmix__2022-03-23_10-59-42" # bruno
    model_id = "qmix__2022-03-23_11-47-25" # warehouse-midd`le
    model_id = "qmix__2022-03-24_00-24-47"   
    model_id = "qmix__2022-03-23_18-35-08" 
    model_id = "qmix__2022-03-28_13-47-25"
    model_id = "qmix__2022-03-29_10-19-01"
    
    #run_per(model_id)
    model_id = "qmix__2022-03-30_11-03-11"
    model_id = "qmix__2022-03-29_10-19-01"
    run_regular(model_id)
    
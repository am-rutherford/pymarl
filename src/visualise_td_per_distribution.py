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
        per_data[t] = th.load(os.path.join(model_path, "per_objs.th"), map_location=th.device('cpu'))
    return per_data, timesteps

def run_td_per(model_id):
    """ for current format """
    
    def _process_buffer(_per_data, _buffer_size):
        data = {}
        
        data["rsum_record"] = th.stack(list(_per_data["reward_sum_record"].values())).cpu().detach().numpy().flatten()
        data["ecount"] = len(data["rsum_record"])
        
        if data["ecount"] > _buffer_size:
            offset = len(_per_data["reward_sum_record"]) % _buffer_size
        
            data["td_errors_record"] = np.concatenate((_per_data["td_errors"][offset:], _per_data["td_errors"][:offset]))  # order pvalues to line up with reward_sum_record
            x_offset = len(_per_data["reward_sum_record"]) - _buffer_size
            data["pval_recordx"] = np.arange(x_offset, x_offset + _buffer_size)
            
            data["e_in_buffer"] = _buffer_size
        else:
            x_offset = 0
            data["td_errors_record"] = _per_data["td_errors"][:data["ecount"]]
            data["e_in_buffer"] = data["ecount"]
            
        data["pval_recordx"] = np.arange(x_offset, x_offset + _buffer_size)
        
        data["td_errors"] = _per_data["td_errors"]
        data["td_errors"][data["e_in_buffer"]:] = 0  # set unfilled to 0
        data["buffer_sample_count"] = _per_data["buffer_sample_count"]
        data["sample_count"] = _per_data["sample_count"]
        data["buffer_sortedidx"] = np.argsort(data["td_errors"][:data["e_in_buffer"]])
        #print(f'len buffer sample count {len(data["buffer_sample_count"])}, len sample count {len(data["sample_count"])}, len r sum record {len(data["rsum_record"])}')
        return data
    
    def _movingaverage(interval, window_size=100):
        window = np.ones(int(window_size))/float(window_size)
        return np.convolve(interval, window, 'same')
    
    buffer_size = 5000
    
    ## --- Load data ---
    per_data, timesteps = load_per_objs(model_id)
    per_proccessed = {t: _process_buffer(per_data[t], buffer_size) for t in timesteps}
    t_max = max(timesteps)
    print('per_data loaded. Keys:', per_data[t_max].keys())# ' epsiodes:', len(per_data[t_max]["reward_sum_record"]))
    ## --- Figure 1: Buffer statistics
    #fig = go.Figure()
    fig = make_subplots(3, 2, subplot_titles=("Plot 1", "Plot 2", "Plot 3", "Plot 4", "Plot 4", "Plot 4"))
    
    # Add traces, one for each slider step     
    for t in timesteps:
        sorted_indicies = per_proccessed[t]["buffer_sortedidx"]
        fig.add_trace(
            go.Scatter(
                visible=False,
                line=dict(color="#00CED1", width=1),
                name="timestep: " + str(t),
                y=per_data[t]["td_errors"][sorted_indicies],
            ),
            row=1,
            col=1,
        )
        npv = per_data[t]["td_errors"][sorted_indicies]
        fig.add_trace(
            go.Scatter(
                visible=False,
                line=dict(color="#d12600", width=1),
                name="timestep: " + str(t),
                y=npv/np.sum(npv),#/np.sum(per_data[t]["pvalues"][sorted_indicies]),
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
            row=3,
            col=1,
        )
        
        fig.add_trace(
            go.Scatter(
                visible=False,
                line=dict(color="#00CED1", width=1),
                name="timestep: " + str(t),
                y=per_proccessed[t]["rsum_record"],
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                visible=False,
                line=dict(color="#d12600", width=1),
                name="timestep: " + str(t),
                y=[1,1],
            ),
            row=2,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                visible=False,
                line=dict(color="#d12600", width=1),
                name="timestep: " + str(t),
                y=per_proccessed[t]["buffer_sample_count"],
            ),
            row=3,
            col=2,
        )
        
        
    # Make 1st trace visible
    c = 6
    for d in range(c):
        fig.data[d].visible = True

    # Create and add slider
    steps = []
    for i in range(len(timesteps)):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)},
                {"title": "Slider switched to timestep: " + str(timesteps[i])}],  # layout attribute
        )
        for d in range(c):
            step["args"][0]["visible"][c*i+d] = True  # Toggle i'th trace to "visible"
            #step["args"][0]["visible"][c*i+1] = True
            #step["args"][0]["visible"][c*i+2] = True
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
    rmax = max([max(per_data[t]["td_errors"]) for t in timesteps])*1.1
    pmax = max([max(per_data[t]["td_errors"][per_proccessed[t]["buffer_sortedidx"]]/np.sum(per_data[t]["td_errors"][per_proccessed[t]["buffer_sortedidx"]])) for t in timesteps])*1.1
    print('r range', [0, (per_proccessed[t_max]["rsum_record"])])
    fig.update_yaxes(title_text="yaxis 1 title", range=[0, rmax], row=1, col=1)
    fig.update_yaxes(title_text="yaxis 1 title", range=[0, rmax], row=1, col=2)
    #fig.update_yaxes(title_text="yaxis 2 title", range=[0, pmax], row=2, col=1)
    fig.update_yaxes(title_text="yaxis 2 title", range=[0, pmax], row=2, col=2)
    #fig.update_yaxes(title_text="yaxis 3 title", range=[0, max(list(per_data[t_max]["sample_count"].values()))*1.1], row=3, col=1)
    #fig.update_yaxes(title_text="yaxis 3 title", range=[0, max(list(per_data[t_max]["sample_count"].values()))*1.1], row=3, col=2)
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
    


if __name__ == "__main__":
    model_id = "qmix__2022-04-06_13-11-00"
    run_td_per(model_id)
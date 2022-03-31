from collections import defaultdict
import logging
import numpy as np
import torch as th

class Logger:
    def __init__(self, console_logger):
        self.console_logger = console_logger

        self.use_tb = False
        self.use_sacred = False
        self.use_hdf = False

        self.stats = defaultdict(lambda: [])

    def setup_tb(self, directory_name):
        # Import here so it doesn't have to be installed if you don't use it
        from tensorboard_logger import configure, log_value, log_histogram
        configure(directory_name)
        self.tb_logger = log_value
        self.tb_logger_hist = log_histogram
        self.use_tb = True

    def setup_sacred(self, sacred_run_dict):
        self.sacred_info = sacred_run_dict.info
        self.use_sacred = True

    def log_stat(self, key, value, t, to_sacred=True):
        self.stats[key].append((t, value))

        if self.use_tb:
            self.tb_logger(key, value, t)

        if self.use_sacred and to_sacred:
            if key in self.sacred_info:
                self.sacred_info["{}_T".format(key)].append(t)
                self.sacred_info[key].append(value)
            else:
                self.sacred_info["{}_T".format(key)] = [t]
                self.sacred_info[key] = [value]

    def print_recent_stats(self):
        log_str = "Recent Stats | t_env: {:>10} | Episode: {:>8}\n".format(*self.stats["episode"][-1])
        i = 0
        for (k, v) in sorted(self.stats.items()):
            if k == "episode":
                continue
            i += 1
            window = 5 if k != "epsilon" else 1
            item = "{:.4f}".format(np.mean([x[1].cpu() if th.is_tensor(x[1]) else x[1] for x in self.stats[k][-window:]]))
            log_str += "{:<25}{:>8}".format(k + ":", item)
            log_str += "\n" if i % 4 == 0 else "\t"
        self.console_logger.info(log_str)


# set up a custom logger
def get_logger():
    logger = logging.getLogger()
    logger.handlers = []
    ch = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s %(asctime)s] %(name)s %(message)s', '%H:%M:%S')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel('DEBUG')

    return logger


def log_mac_weights(logger, mac, idx):
    logger.tb_logger_hist("agent fc1 weight", mac.agent.fc1.weight.detach().numpy(), idx)
    logger.tb_logger_hist("agent fc1 bias", mac.agent.fc1.bias.detach().numpy(), idx)
    logger.tb_logger_hist("agent gru ih weight", mac.agent.rnn.weight_ih.detach().numpy(), idx)
    logger.tb_logger_hist("agent gru ih bias", mac.agent.rnn.bias_ih.detach().numpy(), idx)
    logger.tb_logger_hist("agent gru hh weight", mac.agent.rnn.weight_hh.detach().numpy(), idx)
    logger.tb_logger_hist("agent gru hh bias", mac.agent.rnn.bias_hh.detach().numpy(), idx)
    logger.tb_logger_hist("agent fc2 weight", mac.agent.fc2.weight.detach().numpy(), idx)
    logger.tb_logger_hist("agent fc2 bias", mac.agent.fc2.bias.detach().numpy(), idx)   

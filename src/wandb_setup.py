"""
configuring wandb settings
"""
import os
from copy import deepcopy
from src.config_load import update_cfg, cfgnode_to_dict, recursive_search_key, flatten_cfg_node
import wandb
from yacs.config import CfgNode as CN

def initialise_wandb(args_str, config=None):
    opt = deepcopy(args_str)
    opt = cfgnode_to_dict(opt)

    if config:
        opt.update(config)
    if opt['wenable']:
        if 'use_wandb_offline' in opt.keys():
            os.environ["WANDB_MODE"] = "offline"
        else:
            os.environ["WANDB_MODE"] = "run"
        if 'wandb_run_name' in opt.keys():
            wandb.init(project=opt['wandb_project'], config=opt) #, group=opt['wandb_group'],
                    #   name=opt['wandb_run_name'], reinit=True, config=opt
        else:
            wandb.init(entity=opt['wandb_entity'], project=opt['wandb_project'], config=opt)

        wandb.define_metric("epoch_step")  # Customize axes - https://docs.wandb.ai/guides/track/log
        if 'wandb_track_grad_flow' in opt:
            wandb.define_metric("grad_flow_step")  # Customize axes - https://docs.wandb.ai/guides/track/log
            wandb.define_metric("gf_e*", step_metric="grad_flow_step")  # grad_flow_epoch*

        return wandb.config  # access all HPs through wandb.config, so logging matches execution!

    else:
        os.environ["WANDB_MODE"] = "disabled"  # sets as NOOP, saves keep writing: if opt['wandb']:
        return args_str

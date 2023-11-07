"""
configuring wandb settings
"""
import os
from copy import deepcopy
from src.config_load import update_cfg, cfgnode_to_dict, recursive_search_key
import wandb


def initialise_wandb(args, config=None):
    a = deepcopy(cfgnode_to_dict(args))
    if config:
        a.update(config)
    if 'wandb' in a:
        opt = deepcopy(a['wandb'])
        if opt['use_wandb_offline']:
            os.environ["WANDB_MODE"] = "offline"
        else:
            os.environ["WANDB_MODE"] = "run"
        if 'wandb_run_name' in opt.keys():
            wandb.init(project=opt['wandb_project'], config=opt, group=opt['wandb_group'],
                      name=opt['wandb_run_name'], reinit=True)
        else:
            wandb.init(entity=opt['wandb_entity'], project=opt['wandb_project'], config=opt) # , group=opt['wandb_group'],
                    #   reinit=True, 

        wandb.define_metric("epoch_step")  # Customize axes - https://docs.wandb.ai/guides/track/log
        if opt['wandb_track_grad_flow']:
            wandb.define_metric("grad_flow_step")  # Customize axes - https://docs.wandb.ai/guides/track/log
            wandb.define_metric("gf_e*", step_metric="grad_flow_step")  # grad_flow_epoch*

        return wandb.config  # access all HPs through wandb.config, so logging matches execution!

    else:
        os.environ["WANDB_MODE"] = "disabled"  # sets as NOOP, saves keep writing: if opt['wandb']:
        return 

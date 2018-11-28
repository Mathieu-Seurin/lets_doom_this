import json
import os

import numpy as np
from env.doom_tool import vizdoom_basic_creator
from ray.tune import register_env, Experiment

import policy_model

from ray.rllib.agents import ppo, dqn
from ray.tune import function as call_back_function
from env.callback import on_episode_start, on_episode_end, on_episode_step, on_sample_end

def override_config_recurs(config, config_extension):
    try:
        config_extension['name'] = config['name']+'_'+config_extension['name']
    except KeyError:
        pass

    for key, value in config_extension.items():
        if type(value) is dict:
            config[key] = override_config_recurs(config[key], config_extension[key])
        else:
            assert key in config, "Warning, key defined in extension but not original : key is {}".format(key)
            config[key] = value

    return config

def load_single_config(config_file):
    with open(config_file, 'rb') as f_config:
        config_str = f_config.read().decode('utf-8')
        config = json.loads(config_str)
    return config

def create_expe_spec(config, n_cpu, n_gpu, exp_dir):

    # Create env and register it, so ray and rllib can use it
    register_env(config["env_config"]["env"], lambda env_config: vizdoom_basic_creator(env_config))

    # Expe config, it's a mess to makes things work
    # Lots of things hidden here, check .json in config/env to see more info
    expe_config = {}
    expe_config["env"] = config["env_config"]["env"]
    expe_config["env_config"] = config["env_config"].copy() # env config
    expe_config.update(config["algo_config"].copy()) # algo + model config

    #expe_config["optimizer"] = {"debug" : True}

    expe_config["callbacks"] = {
        # "on_episode_start": call_back_function(on_episode_start),
        #"on_episode_step": call_back_function(on_episode_step),
        #"on_episode_end": call_back_function(on_episode_end),
        #"on_sample_end": call_back_function(on_sample_end),
        }

    trial_resources = {"cpu": n_cpu,"gpu": n_gpu}


    experiment = Experiment(name=expe_config["env"],
                            run=config["algo"],
                            stop=config["stop"],
                            config=expe_config,
                            trial_resources=trial_resources,
                            num_samples=1,
                            checkpoint_freq= 2,
                            max_failures=2,
                            )
    return experiment

def load_config_and_ext(env_config_file, model_config_file, seed,
                        args=None,
                        env_ext_file=None,
                        model_ext_file=None,
                        ):

    # Load env file and model
    env_config = load_single_config(os.path.join("config","env",env_config_file))
    model_config = load_single_config(os.path.join("config","model",model_config_file))

    # Override model file if specified
    if model_ext_file:
        model_ext_config = load_single_config(os.path.join("config","model_ext",model_ext_file))
        model_config = override_config_recurs(model_config, model_ext_config)

    # Override env file if specified
    if env_ext_file:
        model_ext_config = load_single_config(os.path.join("config", "env_ext", env_ext_file))
        model_config = override_config_recurs(model_config, model_ext_config)

    # Merge env and model config into one dict

    full_config = {**model_config, **env_config}

    # set seed
    #set_seed(seed)

    return full_config


def set_seed(seed):

    import torch
    import random
    import tensorflow
    import numpy as np

    if seed > -1:
        print('Using seed {}'.format(seed))
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        tensorflow.set_random_seed(seed)
    else:
        raise NotImplementedError("Cannot set negative seed")


def select_agent(config):

    if config["algo"].lower() == "ppo":
        agent = ppo.PPOAgent(config=config["algo_config"],
                             env=config["env_config"]["env"])

    elif config["algo"].lower() == "apex":
        agent = dqn.ApexAgent(config=config["algo_config"],
                             env=config["env_config"]["env"])
    else:
        raise NotImplementedError("PPO and Apex are available, that's all")

    return agent


if __name__ == "__main__":
    pass

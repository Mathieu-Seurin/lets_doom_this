import json
import os

from env.doom_tool import vizdoom_basic_creator
from ray.tune import register_env, Experiment
from ray.tune.schedulers import pbt

from ray.rllib.agents import ppo, dqn
import random


def override_config_recurs(config, config_extension):

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


def merge_env_algo_config(config):
    def _check_config_spec(expe_config):
        # if embedding size == 0, use one-hot vector for env
        if expe_config["model"]["custom_options"]["text_objective_config"]["embedding_size"] == 0:
            expe_config["env_config"]["onehot"] = True
            # todo : super ugly.
            # But cannot use preprocessor because of multiple intputs

        return expe_config

    # Expe config, it's a mess to makes things work
    # Lots of things hidden here, check .json in config/env to see more info
    expe_config = {}
    expe_config["env"] = config["env_config"]["env"]
    expe_config["env_config"] = config["env_config"].copy()  # env config
    expe_config.update(config["algo_config"].copy())  # algo + model config

    # expe_config["optimizer"] = {"debug" : True}

    expe_config["callbacks"] = {
        # "on_episode_start": call_back_function(on_episode_start),
        # "on_episode_step": call_back_function(on_episode_step),
        # "on_episode_end": call_back_function(on_episode_end),
        # "on_sample_end": call_back_function(on_sample_end),
    }

    return _check_config_spec(expe_config)


def create_expe_spec(config, n_cpu, n_gpu, exp_dir):

    # Create env and register it, so ray and rllib can use it
    register_env(config["env_config"]["env"], lambda env_config: vizdoom_basic_creator(env_config))

    expe_config = merge_env_algo_config(config)

    trial_resources = {"cpu": n_cpu, "gpu": n_gpu}

    experiment = Experiment(name=config["name_expe"],
                            run=config["algo"],
                            stop=config["stop"],
                            config=expe_config,
                            trial_resources=trial_resources,
                            num_samples=1,
                            checkpoint_freq= 2,
                            max_failures=2,
                            local_dir=exp_dir)

    return experiment

def load_config(env_config_file, model_config_file,
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

    algo = config["algo"].lower()
    env =  config["env_config"]["env"]
    expe_config = merge_env_algo_config(config)

    if algo == "ppo":
        agent = ppo.PPOAgent(config=expe_config,
                             env=env)

    elif algo == "apex":
        agent = dqn.ApexAgent(config=expe_config,
                             env=env)
    else:
        raise NotImplementedError("PPO and Apex are available, that's all")

    return agent

def prepare_pbt_config(expe_config, pbt_config):
    """
    todo : recursive when it's done in rllib
    """
    def explore(config):

        # Ensure we update the network
        if config["target_network_update_freq"] < 5 :
            config["target_network_update_freq"] = 5

        # Ensure batch_size not too big, to avoid GPU memory overload
        if config["train_batch_size"] > 1500:
            config["num_sgd_iter"] = 1500

        return config

    pbt_config = load_single_config(os.path.join("config","pbt",pbt_config))

    # Change experiment name, to try few things
    expe_config["name"] += "_pbt_test"

    # Override base config with random params from pbt config file
    for param_key, value in pbt_config["init_override"].items():
        expe_config[param_key] = lambda: random.choice(value)


    #Prepare parameters mutations for pbt
    pbt_parameters_mutations = {}

    # create config, parameters by parameters
    for param_key, param_dict in pbt_config["hyperparam_mutations"].items():

        # Choice = choose among a list of params
        if param_dict["choice"] :
            pbt_parameters_mutations[param_key] = param_dict["range"]
        else: # Sample uniformly, min and max given
            assert len(param_dict["range"]) == 2, "Should be a list of len 2 (is {}) [min_range, max_range]"

            # Check type -> if int : random.randint
            if type(param_dict["range"][0]) is int :
                pbt_parameters_mutations[param_key] = lambda: random.randint(*param_dict["range"])
            elif type(param_dict["range"][0]) is float :
                pbt_parameters_mutations[param_key] = lambda: random.uniform(*param_dict["range"])

        # Configure pbt config
    pbt_scheduler = pbt.PopulationBasedTraining(
        time_attr=pbt_config["time_attr"],
        reward_attr=pbt_config["reward_attr"],
        perturbation_interval=pbt_config["perturbation_interval"],
        resample_probability=pbt_config["resample_probability"],
        hyperparam_mutations=pbt_parameters_mutations,
        custom_explore_fn=explore
    )

    return expe_config, pbt_scheduler



if __name__ == "__main__":
    pass

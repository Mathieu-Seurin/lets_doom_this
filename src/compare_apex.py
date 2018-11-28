from ray.rllib.agents import ppo, dqn
import argparse
import ray

from config import load_config_and_ext, create_expe_spec, select_agent

from env.doom_tool import vizdoom_basic_creator
from ray.tune import register_env

from ray.tune.logger import pretty_print
from ray.tune import function as call_back_function
from env.callback import on_episode_end

import gym

def vizdoom_basic_creator(env_config):
    import gym_vizdoom

    env_name = "VizdoomBasic-v0"
    #env_name = env_config["env"] + "-v0"

    return gym.make(env_name)

env = "VizdoomBasic"
register_env("VizdoomBasic", lambda env_config: vizdoom_basic_creator(env_config))

# env = "cartpole"
# register_env(env, lambda env_config: gym.make("CartPole-v0"))

ray.init()
algo_config = {
    "num_envs_per_worker": 2,
    "num_workers": 24,
    "num_gpus": 1,
    "gamma": 0.95,

    "lr": 1e-4,
    "n_step": 3,

    "collect_metrics_timeout": 10,

    "learning_starts": 100,
    "train_batch_size": 1024,
    "target_network_update_freq": 1000,
    "timesteps_per_iteration": 1000,

    "model": {
        "custom_model": None,
        "custom_preprocessor": None,
        "custom_options": {
        }

    }
}

agent = dqn.ApexAgent(config=algo_config,
                      env=env)

for i in range(2):
   result = agent.train()
   print(pretty_print(result))

agent = None

expe_config = {}
expe_config.update(algo_config)
expe_config["env"] = env
expe_config["env_config"] = {}
expe_config["callbacks"] = {"on_episode_end": call_back_function(on_episode_end)}


stop = {"episodes_total" : 2}
trial_resources = {"cpu": 24, "gpu": 1}

experiment = ray.tune.Experiment(name="test",
                            run="APEX",
                            stop=stop,
                            config=expe_config,
                            trial_resources=trial_resources,
                            num_samples=1,
                            checkpoint_freq= 2,
                            max_failures=2,
                            local_dir="out/"
                            )

ray.tune.run_experiments(experiments=experiment)
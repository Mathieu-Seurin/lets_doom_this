from ray.rllib.agents import ppo, dqn
import argparse
import ray

from config import load_config, create_expe_spec, select_agent

from env.doom_tool import vizdoom_basic_creator
from ray.tune import register_env

from ray.tune.logger import pretty_print
from ray.tune import function as call_back_function
from env.callback import on_episode_end
from neural_toolbox import policy_model

parser = argparse.ArgumentParser('Log Parser arguments!')

parser.add_argument("-env_config",   type=str)
parser.add_argument("-env_ext",      type=str)
parser.add_argument("-model_config", type=str)
parser.add_argument("-model_ext",    type=str)
parser.add_argument("-exp_dir",      type=str, default="out", help="Directory all results")
parser.add_argument("-seed",         type=int, default=0, help="Random seed used")
parser.add_argument("-n_cpu",        type=int, default=24, help="How many cpus do you want ?")
parser.add_argument("-n_gpu",        type=int, default=2, help="How many gpus do you want ? Because we own too many")


args = parser.parse_args()

ray.init(object_store_memory=int(6e10),
         num_cpus=args.n_cpu,
         num_gpus=args.n_gpu,
         )


full_config = load_config(env_config_file=args.env_config,
                          model_config_file=args.model_config,
                          env_ext_file=args.env_ext,
                          model_ext_file=args.model_ext
                          )


register_env(full_config["env_config"]["env"], lambda env_config: vizdoom_basic_creator(env_config))


#full_config["callbacks"] = {"on_episode_end" : call_back_function(on_episode_end)}
full_config["algo_config"]["monitor"] = False

agent = select_agent(full_config)

for i in range(3000):
   result = agent.train()
   print(pretty_print(result))

   if i % 10 == 0:
       checkpoint = agent.save()
       print("checkpoint saved at", checkpoint)

import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

import argparse

import ray

from config import load_config, create_expe_spec, set_seed, prepare_pbt_config
from neural_toolbox import policy_model

parser = argparse.ArgumentParser('Log Parser arguments!')

parser.add_argument("-env_config",   type=str)
parser.add_argument("-env_ext",      type=str)
parser.add_argument("-model_config", type=str)
parser.add_argument("-model_ext",    type=str)
parser.add_argument("-pbt_config",   type=str, help="json file with population base training parameters")
parser.add_argument("-exp_dir",      type=str, default="out", help="Directory all results")
parser.add_argument("-seed",         type=int, default=0, help="Random seed used")
parser.add_argument("-n_cpu",        type=int, default=24, help="How many cpus do you want ?")
parser.add_argument("-n_gpu",        type=int, default=2, help="How many gpus do you want ? Because we own too many")


args = parser.parse_args()


set_seed(args.seed)
ray.init(object_id_seed=args.seed, num_gpus=args.n_gpu, num_cpus=args.n_cpu)


full_config = load_config(env_config_file=args.env_config,
                          model_config_file=args.model_config,
                          env_ext_file=args.env_ext,
                          model_ext_file=args.model_ext)

if args.pbt_config:
    full_config, pbt_scheduler = prepare_pbt_config(full_config, args.pbt_config)
else:
    pbt_scheduler = None

experiment = create_expe_spec(full_config,
                              n_cpu=args.n_cpu,
                              n_gpu=args.n_gpu,
                              exp_dir=args.exp_dir)


ray.tune.run_experiments(experiments=experiment,
                         scheduler=pbt_scheduler,
                         queue_trials=True)
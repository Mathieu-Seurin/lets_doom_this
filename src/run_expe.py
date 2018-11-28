import argparse
import ray

from config import load_config_and_ext, create_expe_spec

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

set_seed(args.seed)
ray.init(object_id_seed=args.seed)



full_config = load_config_and_ext(env_config_file=args.env_config,
                                  model_config_file=args.model_config,
                                  seed=args.seed,

                                  env_ext_file=args.env_ext,
                                  model_ext_file=args.model_ext
                                  )

experiment = create_expe_spec(full_config,
                                        n_cpu=args.n_cpu,
                                        n_gpu=args.n_gpu,
                                        exp_dir=args.exp_dir)


ray.tune.run_experiments(experiment)

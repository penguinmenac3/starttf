import sys
from setproctitle import setproctitle
import tensorflow as tf

from hyperparams.hyperparams import import_params, load_params

if tf.__version__.startswith("1."):
    print("Using keras for tensorflow 1.x")
    from starttf.train.keras import easy_train_and_evaluate
else:
    from starttf.train.supervised import easy_train_and_evaluate


def main(args):
    if len(args) == 2 or len(args) == 3:
        continue_training = False
        no_artifacts = False
        idx = 1
        if args[idx] == "--continue":
            continue_training = True
            idx += 1
        if args[idx] == "--no_artifacts":
            no_artifacts = True
            idx += 1
        if args[1].endswith(".json"):
            hyperparams = load_params(args[idx])
        elif args[1].endswith(".py"):
            hyperparams = import_params(args[idx])
        name = hyperparams.train.get("experiment_name", "unnamed")
        setproctitle("train {}".format(name))
        return easy_train_and_evaluate(hyperparams, continue_training=continue_training, no_artifacts=no_artifacts)
    else:
        print("Usage: python -m starttf.train [--continue] hyperparameters/myparams.py")
        return None

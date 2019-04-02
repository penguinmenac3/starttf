import sys
import tensorflow as tf

from hyperparams.hyperparams import load_params

if tf.__version__.startswith("1."):
    print("Using keras for tensorflow 1.x")
    from starttf.train.keras import easy_train_and_evaluate
else:
    from starttf.train.supervised import easy_train_and_evaluate

if len(sys.argv) == 2 or len(sys.argv) == 3:
    continue_training = False
    idx = 1
    if sys.argv[idx] == "--continue":
        continue_training = True
        idx += 1
    hyperparams = load_params(sys.argv[1])
    easy_train_and_evaluate(hyperparams, continue_training=continue_training)
else:
    print("Usage: python -m starttf.train [--continue] hyperparameters/myparams.json")

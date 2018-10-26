# MIT License
# 
# Copyright (c) 2018 Michael Fuerst
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import time
import datetime
import json

import tensorflow as tf

from starttf.utils.plot_losses import DefaultLossCallback
from starttf.utils.session_config import get_default_config

from starttf.data.autorecords import create_input_fn, PHASE_TRAIN, PHASE_VALIDATION
from starttf.utils.create_optimizer import create_optimizer


def create_tf_estimator_spec(chkpt_path, Model, create_loss, inline_plotting=False):
    report_storage = {}

    def my_model_fn(features, labels, mode, params):
        is_mode_training = mode == tf.estimator.ModeKeys.TRAIN

        # Create a model
        model = Model(params)
        tf_model = model.create_tf_model(features, training=is_mode_training)

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode, predictions=tf_model)

        # Add a loss
        losses, metrics = create_loss(tf_model, labels, mode, params)
        loss = losses["loss"]

        for k in losses.keys():
            tf.summary.scalar(k, losses[k])

        if mode == tf.estimator.ModeKeys.EVAL:
            hooks = [DefaultLossCallback(params, losses, chkpt_path + "/eval", mode="eval",
                                         report_storage=report_storage, inline_plotting=inline_plotting)]
            return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics, evaluation_hooks=hooks)

        with tf.variable_scope("optimizer"):
            # Define a training operation
            if mode == tf.estimator.ModeKeys.TRAIN:
                optimizer, global_step = create_optimizer(params)
                train_op = optimizer.minimize(loss, global_step=global_step)

                hooks = [DefaultLossCallback(params, losses, chkpt_path, mode="train",
                                             report_storage=report_storage, inline_plotting=inline_plotting)]
                return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, training_hooks=hooks)

        raise RuntimeError("Unexpected mode.")

    return my_model_fn


def easy_train_and_evaluate(hyper_params, Model, create_loss,
                            training_data=None, validation_data=None,
                            inline_plotting=False, session_config=None, log_suffix=None, 
                            continue_training=False, continue_with_specific_checkpointpath=None):
    """
    Train and evaluate your model without any boilerplate code.

    1) Write your data using the starttf.tfrecords.autorecords.write_data method.
    2) Create your hyper parameter file containing all required fields and then load it using
        starttf.utils.hyper_params.load_params method.
        Minimal Sample Hyperparams File:
        {"train": {
            "learning_rate": {
                "type": "const",
                "start_value": 0.001
            },
            "optimizer": {
                "type": "adam"
            },
            "batch_size": 1024,
            "iters": 10000,
            "summary_iters": 100,
            "checkpoint_path": "checkpoints/mnist",
            "tf_records_path": "data/.records/mnist"
            }
        }
    3) Pass everything required to this method and that's it.
    :param hyper_params: The hyper parameters obejct loaded via starttf.utils.hyper_params.load_params
    :param Model: A keras model.
    :param create_loss: A create_loss function like that in starttf.examples.mnist.loss.
    :param inline_plotting: When you are using jupyter notebooks you can tell it to plot the loss directly inside the notebook.
    :param continue_training: Bool, continue last training in the checkpoint path specified in the hyper parameters.
    :param session_config: A configuration for the session.
    :param log_suffix: A suffix for the log folder, so you can remember what was special about the run.
    :return:
    """
    time_stamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H.%M.%S')
    chkpt_path = hyper_params.train.checkpoint_path + "/" + time_stamp
    if log_suffix is not None:
        chkpt_path = chkpt_path + "_" + log_suffix

    if session_config is None:
        session_config = get_default_config()

    if continue_with_specific_checkpointpath:
        chkpt_path = hyper_params.train.checkpoint_path + "/" + continue_with_specific_checkpointpath
        print("Continue with checkpoint: {}".format(chkpt_path))
    elif continue_training:
        chkpts = sorted([name for name in os.listdir(hyper_params.train.checkpoint_path)])
        chkpt_path = hyper_params.train.checkpoint_path + "/" + chkpts[-1]
        print("Latest found checkpoint: {}".format(chkpt_path))

    if not os.path.exists(chkpt_path):
        os.makedirs(chkpt_path)

    # Load training data
    print("Load data")
    if training_data is None:
        training_data = create_input_fn(os.path.join(hyper_params.train.tf_records_path, PHASE_TRAIN),
                                        hyper_params.train.batch_size)
    if validation_data is None:
        validation_data = create_input_fn(os.path.join(hyper_params.train.tf_records_path, PHASE_VALIDATION),
                                          hyper_params.train.batch_size)

    # Write hyper parameters to be able to track what config you had.
    with open(chkpt_path + "/hyperparameters.json", "w") as json_file:
        json_file.write(json.dumps(hyper_params.to_dict(), indent=4, sort_keys=True))

    estimator_spec = create_tf_estimator_spec(chkpt_path, Model, create_loss, inline_plotting)

    # Create a run configuration
    config = None
    if hyper_params.train.get("distributed", False):
        distribution = tf.contrib.distribute.MirroredStrategy()
        config = tf.estimator.RunConfig(model_dir=chkpt_path,
                                        save_summary_steps=hyper_params.train.summary_steps,
                                        train_distribute=distribution,
                                        save_checkpoints_steps=hyper_params.train.save_checkpoint_steps,
                                        keep_checkpoint_max=hyper_params.train.keep_checkpoint_max,
                                        keep_checkpoint_every_n_hours=1)
    else:
        config = tf.estimator.RunConfig(session_config=session_config,
                                        model_dir=chkpt_path,
                                        save_summary_steps=hyper_params.train.summary_steps,
                                        save_checkpoints_steps=hyper_params.train.save_checkpoint_steps,
                                        keep_checkpoint_max=hyper_params.train.keep_checkpoint_max,
                                        keep_checkpoint_every_n_hours=1)

    # Create the estimator.
    estimator = None
    if hyper_params.train.get("warm_start_checkpoint", None) is not None:
        warm_start_dir = hyper_params.train.warm_start_checkpoint
        estimator = tf.estimator.Estimator(estimator_spec,
                                           config=config,
                                           warm_start_from=warm_start_dir,
                                           params=hyper_params)
    else:
        estimator = tf.estimator.Estimator(estimator_spec,
                                           config=config,
                                           params=hyper_params)

    # Specify training and actually train.
    throttle_secs = hyper_params.train.get("throttle_secs", 120)
    train_spec = tf.estimator.TrainSpec(input_fn=training_data,
                                        max_steps=hyper_params.train.steps)
    eval_spec = tf.estimator.EvalSpec(input_fn=validation_data,
                                      throttle_secs=throttle_secs)

    print("Start training")
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    return estimator


def create_prediction_estimator(hyper_params, model, checkpoint_path=None):
    """
    Create an estimator for prediction purpose only.
    :param hyper_params: The hyper params file.
    :param model: The keras model.
    :param checkpoint_path: (Optional) Path to the specific checkpoint to use.
    :return:
    """
    if checkpoint_path is None:
        chkpts = sorted([name for name in os.listdir(hyper_params.train.checkpoint_path)])
        checkpoint_path = hyper_params.train.checkpoint_path + "/" + chkpts[-1]
        print("Latest found checkpoint: {}".format(checkpoint_path))

    estimator_spec = create_tf_estimator_spec(checkpoint_path, model, create_loss=None)

    # Create the estimator.
    estimator = tf.estimator.Estimator(estimator_spec,
                                       model_dir=checkpoint_path,
                                       params=hyper_params)

    return estimator

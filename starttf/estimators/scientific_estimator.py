import os
import time
import datetime
import json

import tensorflow as tf

from starttf.utils.plot_losses import DefaultLossCallback
from starttf.utils.session_config import get_default_config

from starttf.tfrecords.autorecords import create_input_fn, PHASE_TRAIN, PHASE_VALIDATION


def create_tf_estimator_spec(chkpt_path, create_model, create_loss, inline_plotting=False):
    report_storage = {}

    def my_model_fn(features, labels, mode, params):
        # Create a model
        model = create_model(features, mode, params)
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode, predictions=model)

        # Add a loss
        losses, metrics = create_loss(model, labels, mode, params)
        loss = losses["loss"]
        for k in losses.keys():
            tf.summary.scalar(k, losses[k])

        if mode == tf.estimator.ModeKeys.EVAL:
            hooks = [DefaultLossCallback(params, losses, chkpt_path + "/eval", mode="eval",
                                         report_storage=report_storage, inline_plotting=inline_plotting)]
            return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics, evaluation_hooks=hooks)

        # Define a training operation
        if mode == tf.estimator.ModeKeys.TRAIN:
            learning_rate = None
            global_step = tf.train.get_global_step()
            if params.train.learning_rate.type == "exponential":
                learning_rate = tf.train.exponential_decay(params.train.learning_rate.start_value, global_step,
                                                           params.train.iters,
                                                           params.train.learning_rate.end_value / params.train.learning_rate.start_value,
                                                           staircase=False, name="lr_decay")
                tf.summary.scalar('hyper_params/lr/start_value',
                                  tf.constant(params.train.learning_rate.start_value))
                tf.summary.scalar('hyper_params/lr/end_value', tf.constant(params.train.learning_rate.end_value))
            elif params.train.learning_rate.type == "const":
                learning_rate = tf.constant(params.train.learning_rate.start_value, dtype=tf.float32)
                tf.summary.scalar('hyper_params/lr/start_value',
                                  tf.constant(params.train.learning_rate.start_value))
            else:
                raise RuntimeError("Unknown learning rate: %s" % params.train.learning_rate.type)
            tf.summary.scalar('hyper_params/lr/learning_rate', learning_rate)

            # Setup Optimizer
            train_op = None
            if params.train.optimizer.type == "rmsprop":
                train_op = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(loss,
                                                                                           global_step=global_step)
            elif params.train.optimizer.type == "adam":
                train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss,
                                                                                        global_step=global_step)
            else:
                raise RuntimeError("Unknown optimizer: %s" % params.train.optimizer.type)

            hooks = [DefaultLossCallback(params, losses, chkpt_path, mode="train",
                                         report_storage=report_storage, inline_plotting=inline_plotting)]
            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, training_hooks=hooks)

        raise RuntimeError("Unexpected mode.")

    return my_model_fn


def easy_train_and_evaluate(hyper_params, create_model, create_loss, init_model=None, inline_plotting=False):
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
            "validation_batch_size": 1024,
            "iters": 10000,
            "summary_iters": 100,
            "checkpoint_path": "checkpoints/mnist",
            "tf_records_path": "data/.records/mnist"
            }
        }
    3) Pass everything required to this method and that's it.
    :param hyper_params: The hyper parameters obejct loaded via starttf.utils.hyper_params.load_params
    :param create_model: A create_model function like that in starttf.models.mnist.
    :param create_loss: A create_loss function like that in starttf.examples.mnist.loss.
    :param init_model: A init_model function, if your model gets initialized with pretrained weights (e.g. starttf.models.vgg16_encoder)
    :param inline_plotting: When you are using jupyter notebooks you can tell it to plot the loss directly inside the notebook.
    :return:
    """
    time_stamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H.%M.%S')
    chkpt_path = hyper_params.train.checkpoint_path + "/" + time_stamp

    if not os.path.exists(chkpt_path):
        os.makedirs(chkpt_path)

    # Load training data
    print("Loading data")
    train_dataset = create_input_fn(os.path.join(hyper_params.train.tf_records_path, PHASE_TRAIN),
                                             hyper_params.train.batch_size)
    validation_dataset = create_input_fn(
        os.path.join(hyper_params.train.tf_records_path, PHASE_VALIDATION), hyper_params.train.validation_batch_size)

    # Write hyper parameters to be able to track what config you had.
    with open(chkpt_path + "/hyperparameters.json", "w") as json_file:
        json_file.write(json.dumps(hyper_params.to_dict(), indent=4, sort_keys=True))

    estimator_spec = create_tf_estimator_spec(chkpt_path, create_model, create_loss, inline_plotting)

    # Train model.

    warm_start_dir = None
    if "warm_start_checkpoint" in hyper_params.train.__dict__:
        warm_start_dir = hyper_params.train.warm_start_checkpoint

    config = None
    if "distributed" in hyper_params.train.__dict__ and hyper_params.train.distributed:
        distribution = tf.contrib.distribute.MirroredStrategy()
        config = tf.estimator.RunConfig(model_dir=chkpt_path,
                                        save_summary_steps=hyper_params.train.summary_steps,
                                        train_distribute=distribution,
                                        save_checkpoints_steps=hyper_params.train.save_checkpoint_steps,
                                        keep_checkpoint_max=hyper_params.train.keep_checkpoint_max,
                                        keep_checkpoint_every_n_hours=1)
    else:
        config = tf.estimator.RunConfig(model_dir=chkpt_path,
                                        save_summary_steps=hyper_params.train.summary_steps,
                                        save_checkpoints_steps=hyper_params.train.save_checkpoint_steps,
                                        keep_checkpoint_max=hyper_params.train.keep_checkpoint_max,
                                        keep_checkpoint_every_n_hours=1)
    estimator = tf.estimator.Estimator(estimator_spec,
                                       config=config,
                                       warm_start_from=warm_start_dir,
                                       params=hyper_params)
    train_spec = tf.estimator.TrainSpec(input_fn=train_dataset,
                                        max_steps=hyper_params.train.steps)
    eval_spec = tf.estimator.EvalSpec(input_fn=validation_dataset,
                                      throttle_secs=120)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    return estimator

import os
import time
import datetime
import json

import tensorflow as tf

from starttf.utils.plot_losses import DefaultLossCallback
from starttf.utils.session_config import get_default_config

from starttf.tfrecords.autorecords import read_data, PHASE_TRAIN, PHASE_VALIDATION

TRAIN = tf.estimator.ModeKeys.TRAIN
EVAL = tf.estimator.ModeKeys.EVAL


def train_and_evaluate(hyper_params, session, train_op, metrics=[], callback=None, enable_timing=False):
    """
    Not recommended.

    Use easy_train_and_evaluate instead.

    :param hyper_params: Hyper params object.
    :param session: A session.
    :param train_op: The operation that trains the network.
    :param metrics: Your training and evaluation metrics.
    :param callback: Callbacks that should be called and fed your metrics.
    :param enable_timing: If you want to time execution to know the speed.
    :return:
    """
    time_stamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H.%M.%S')

    # Collect metrics operations.
    metric_ops = []
    for i in range(len(metrics)):
        metric = metrics[i]
        metric_ops += list(metric.values())

    # Init vars.
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    session.run(init_op)

    # Prepare training.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=session)
    saver = tf.train.Saver()

    # Merge all the summaries and write them out
    merged = tf.summary.merge_all()
    log_writer = tf.summary.FileWriter(hyper_params.train.checkpoint_path + "/" + time_stamp, session.graph)
    tf.global_variables_initializer().run()

    # Write hyper parameters to be able to track what config you had.
    with open(hyper_params.train.checkpoint_path + "/" + time_stamp + "/hyperparameters.json", "w") as json_file:
        json_file.write(json.dumps(hyper_params.to_dict(), indent=4, sort_keys=True))

    # Train
    print("Training Model: To reduce overhead no outputs are done. Use tensorboard to see your progress.")
    print("python -m tensorboard.main --logdir=checkpoints")
    last_printout = time.time()
    last_printout_iter = -1
    for i_step in range(hyper_params.train.iters):
        # Train step.
        if i_step % hyper_params.train.summary_iters != 0:
            _ = session.run([train_op])
        else:  # Do validation and summary.
            results = session.run(metric_ops + [merged])
            # Only print summary after some training has happened.
            if i_step > 0:
                if callback is not None:
                    callback(i_step, metrics, results[:-1], hyper_params.train.checkpoint_path + "/" + time_stamp)
                else:
                    print("Iter: %d" % i_step)
                summary = results[-1]
                log_writer.add_summary(summary, i_step)
                saver.save(session, hyper_params.train.checkpoint_path + "/" + time_stamp + "/chkpt",
                           global_step=i_step)

            if enable_timing:
                end = time.time()
                print("Timing: %.3f ms per iteration" % (
                            (end - last_printout) * 1000 / float(i_step - last_printout_iter)))
                last_printout = end
                last_printout_iter = i_step

    saver.save(session, hyper_params.train.checkpoint_path + "/" + time_stamp + "/final_chkpt")
    coord.request_stop()
    coord.join(threads)

    return hyper_params.train.checkpoint_path + "/" + time_stamp


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
    # Load training data
    print("Loading data")
    train_features, train_labels = read_data(os.path.join(hyper_params.train.tf_records_path, PHASE_TRAIN),
                                             hyper_params.train.batch_size)
    validation_features, validation_labels = read_data(
        os.path.join(hyper_params.train.tf_records_path, PHASE_VALIDATION), hyper_params.train.validation_batch_size)

    # Create a training model.
    print("Create training graph")
    train_model = create_model(train_features, TRAIN, hyper_params)
    train_loss, train_metrics = create_loss(train_model, train_labels, TRAIN, hyper_params)

    # Create a validation model.
    print("Create validation graph")
    validation_model = create_model(validation_features, EVAL, hyper_params)
    _, validation_metrics = create_loss(validation_model, validation_labels, EVAL, hyper_params)

    # Setup learning rate
    learning_rate = None
    global_step = tf.Variable(0, trainable=False)
    if hyper_params.train.learning_rate.type == "exponential":
        learning_rate = tf.train.exponential_decay(hyper_params.train.learning_rate.start_value, global_step,
                                                   hyper_params.train.iters,
                                                   hyper_params.train.learning_rate.end_value / hyper_params.train.learning_rate.start_value,
                                                   staircase=False, name="lr_decay")
        tf.summary.scalar('hyper_params/lr/start_value', tf.constant(hyper_params.train.learning_rate.start_value))
        tf.summary.scalar('hyper_params/lr/end_value', tf.constant(hyper_params.train.learning_rate.end_value))
    elif hyper_params.train.learning_rate.type == "const":
        learning_rate = tf.constant(hyper_params.train.learning_rate.start_value, dtype=tf.float32)
        tf.summary.scalar('hyper_params/lr/start_value', tf.constant(hyper_params.train.learning_rate.start_value))
    else:
        raise RuntimeError("Unknown learning rate: %s" % hyper_params.train.learning_rate.type)

    # Setup Optimizer
    train_op = None
    if hyper_params.train.optimizer.type == "rmsprop":
        train_op = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(train_loss, global_step=global_step)
    elif hyper_params.train.optimizer.type == "adam":
        train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(train_loss, global_step=global_step)
    else:
        raise RuntimeError("Unknown optimizer: %s" % hyper_params.train.optimizer.type)

    # Train model.
    print("Start training")
    with tf.Session(config=get_default_config(gpu_memory_usage=0.9, allow_growth=True)) as session:
        if init_model:
            init_model(hyper_params, train_model, session)
        train_and_evaluate(hyper_params, session, train_op,
                           metrics=[train_metrics, validation_metrics],
                           callback=DefaultLossCallback(inline_plotting=inline_plotting).callback,
                           enable_timing=True)

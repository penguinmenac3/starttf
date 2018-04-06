import math
import tensorflow as tf
import numpy as np

from utils.hyperparams import load_params
from utils.plot_losses import create_plot, DefaultLossCallback
from utils.session_config import get_default_config
from utils.generic_data_loader import load_data
from utils.misc import mode_to_str

from datasets.classification.function_generator import function_generator

from tf_models.gru_function_classifier import create_model
from tf_models.model import train, export_graph, load_graph

GENERATE_DATA = False
TRAIN = tf.estimator.ModeKeys.TRAIN
EVAL = tf.estimator.ModeKeys.EVAL
PREDICT = tf.estimator.ModeKeys.PREDICT


def sin_fn(x, off):
    return math.sin(x / 50.0 + off)


def lin_fn(x, off):
    return x / 50.0 + off


def generate_data_fn():
    training_examples_number = 10000
    validation_examples_number = 1000

    train_data = function_generator([sin_fn, lin_fn], 100, training_examples_number)
    validation_data = function_generator([sin_fn, lin_fn], 100, validation_examples_number)
    return train_data, validation_data


def create_loss(model, labels, mode, hyper_params):
    mode_name = mode_to_str(mode)
    metrics = {}

    # Add loss
    labels = tf.reshape(labels, [-1, hyper_params.arch.output_dimension])
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model["logits"], labels=labels))
    tf.summary.scalar(mode_name + '/loss', loss_op)
    metrics[mode_name + '/loss'] = loss_op

    return loss_op, metrics


def main():
    # Define "constants".
    hyper_params = load_params("tf_examples/gru_function_classifier.json")

    # Load training data
    print("Loading data")
    data_tmp_folder = "data/.records/gru_function_classifier"
    train_features, train_labels, validation_features, validation_labels = load_data(hyper_params, generate_data_fn, data_tmp_folder)

    # Create a training model.
    print("Creating Model")
    train_model = create_model(train_features, TRAIN, hyper_params)
    train_loss, train_metrics = create_loss(train_model, train_labels, TRAIN, hyper_params)
    train_op = tf.train.RMSPropOptimizer(learning_rate=hyper_params.train.learning_rate,
                                     decay=hyper_params.train.decay).minimize(train_loss)

    # Create a validation model.
    validation_model = create_model(validation_features, EVAL, hyper_params)
    _, validation_metrics = create_loss(validation_model, validation_labels, EVAL, hyper_params)

    # Create prediction model.
    predict_model = create_model(tf.placeholder(tf.float32, name="data"), PREDICT, hyper_params)
    
    # Train model.
    with tf.Session(config=get_default_config()) as session:
        checkpoint_path = train(hyper_params, session, train_op,
                                metrics=[train_metrics, validation_metrics],
                                callback=DefaultLossCallback().callback,
                                enable_timing=True)

    # Export the trained model
    export_graph(checkpoint_path=checkpoint_path, output_nodes=["GruFunctionClassifier_2/probs"])
    print("Exported to: %s" % checkpoint_path)

    return checkpoint_path


def deploy_test(checkpoint_path):
    hyper_params = load_params("tf_examples/gru_function_classifier.json")

    # Load Graph
    graph = load_graph(checkpoint_path + "/frozen_model.pb", (1, 100, 1), old_input="GruFunctionClassifier_2/Reshape:0")

    # Find relevant tensors
    input_tensor = graph.get_tensor_by_name('input_tensor:0')
    probs = graph.get_tensor_by_name('GruFunctionClassifier_2/probs:0')

    # Create/define inputs
    x = np.array([[[sin_fn(i, 0)] for i in range(100)]], dtype=np.float32)

    with tf.Session(graph=graph) as sess:
        y_out = sess.run(probs, feed_dict={input_tensor: x})
        print(y_out)


if __name__ == "__main__":
    checkpoint_path = main()

    # To check if import works.
    deploy_test(checkpoint_path)

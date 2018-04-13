import tensorflow as tf

print(tf.__version__)

from starttf.utils.hyperparams import load_params
from starttf.utils.plot_losses import DefaultLossCallback
from starttf.utils.session_config import get_default_config
from starttf.utils.generic_data_loader import load_data

from opendatalake.classification.mnist import mnist
from opendatalake.tfrecords import PHASE_TRAIN, PHASE_VALIDATION

from starttf.models.model import train, export_graph
from starttf.models.mnist import create_model
from starttf.losses.mnist import create_loss

GENERATE_DATA = False
TRAIN = tf.estimator.ModeKeys.TRAIN
EVAL = tf.estimator.ModeKeys.EVAL


def generate_data_fn():
    base_dir = "data/mnist"
    train_data = mnist(base_dir=base_dir, phase=PHASE_TRAIN)
    validation_data = mnist(base_dir=base_dir, phase=PHASE_VALIDATION)
    return train_data, validation_data


def main():
    # Load hyper params and training data
    hyper_params = load_params("starttf/examples/mnist.json")

    # Load training data
    print("Loading data")
    data_tmp_folder = "data/.records/mnist"
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
    
    # Train model.
    with tf.Session(config=get_default_config()) as session:
        checkpoint_path = train(hyper_params, session, train_op,
                                metrics=[train_metrics, validation_metrics],
                                callback=DefaultLossCallback().callback,
                                enable_timing=True)

    # Export the trained model
    export_graph(checkpoint_path=checkpoint_path, output_nodes=["MnistNetwork_1/probs"])
    print("Exported to: %s" % checkpoint_path)


if __name__ == "__main__":
    main()

    ## Import a graph
    #graph = load_graph("tf_models/checkpoints/mnist/2018-02-16_02.18.04/frozen_model.pb", (1, 28, 28, 1), old_input="MnistNetwork_1/input:0")

    #for op in graph.get_operations():
    #    print(op.name)

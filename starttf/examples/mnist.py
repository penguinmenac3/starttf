import tensorflow as tf

print(tf.__version__)

from opendatalake.classification.mnist import mnist

from starttf.utils.hyperparams import load_params
from starttf.utils.plot_losses import DefaultLossCallback
from starttf.utils.session_config import get_default_config

from starttf.tfrecords.autorecords import auto_read_write_data, PHASE_TRAIN, PHASE_VALIDATION

from starttf.estimators.scientific_estimator import train_and_evaluate

from starttf.models.utils import export_graph
from starttf.models.mnist import create_model

from starttf.utils.misc import mode_to_str

GENERATE_DATA = False
TRAIN = tf.estimator.ModeKeys.TRAIN
EVAL = tf.estimator.ModeKeys.EVAL


def create_loss(model, labels, mode, hyper_params):
    mode_name = mode_to_str(mode)
    metrics = {}

    # Add loss
    labels = tf.reshape(labels["probs"], [-1, 10])
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model["logits"], labels=labels))
    tf.summary.scalar(mode_name + '/loss', loss_op)
    metrics[mode_name + '/loss'] = loss_op

    return loss_op, metrics


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
    train_features, train_labels, validation_features, validation_labels = auto_read_write_data(hyper_params, generate_data_fn, data_tmp_folder)

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
        checkpoint_path = train_and_evaluate(hyper_params, session, train_op,
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

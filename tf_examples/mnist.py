import tensorflow as tf

from utils.hyperparams import load_params
from utils.plot_losses import create_plot

from datasets.classification.mnist import mnist
from utils.generic_data_loader import load_data
from datasets.tfrecords import PHASE_TRAIN, PHASE_VALIDATION

from tf_models.mnist import create_loss, create_model
from tf_models.model import train, export_graph, load_graph

GENERATE_DATA = False


def generate_data_fn():
    base_dir = "data/mnist"
    train_data = mnist(base_dir=base_dir, phase=PHASE_TRAIN)
    validation_data = mnist(base_dir=base_dir, phase=PHASE_VALIDATION)
    return train_data, validation_data


def main():
    # Load hyper params and training data
    hyper_params = load_params("tf_examples/mnist.json")

    # Load training data
    print("Loading data")
    data_tmp_folder = "data/.records/mnist"
    validation_examples_number = 10000
    train_features, train_labels, validation_features, validation_labels = load_data(hyper_params, generate_data_fn, data_tmp_folder, validation_examples_number=validation_examples_number)

    # Create model.
    print("Creating Model")
    train_model, feed_dict = create_model(hyper_params, train_features)
    validation_model, feed_dict = create_model(hyper_params, validation_features, reuse_weights=True, deploy_model=True,
                                               feed_dict=feed_dict)
    train_op, reports = create_loss(hyper_params, train_model, validation_model, train_labels, validation_labels)

    # Limit used gpu memory.
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.75

    # Collector variables for loss images.
    iter_list = []
    losses = []
    val_losses = []

    # Callback gets called every validation iter.
    def callback(i_step, reports, model_path):
        # Interpret report matching to the report defined in reports by create_model
        print("Iter: %d, Train loss: %.4f, Test loss: %.4f" % (i_step, reports[0], reports[1]))

        # Also plot the loss in a png (e.g. for papers)
        iter_list.append(i_step)
        losses.append(reports[0])
        val_losses.append(reports[1])
        create_plot(model_path, iter_list, losses, val_losses, "Train Loss", "Validation Loss", "Losses")

    # Train model.
    with tf.Session(config=config) as session:
        checkpoint_path = train(hyper_params, session, train_op, feed_dict, reports=reports, callback=callback)

    # Export the trained model
    export_graph(checkpoint_path=checkpoint_path, output_nodes=["MnistNetwork_1/probs"])
    print("Exported to: %s" % checkpoint_path)


if __name__ == "__main__":
    main()

    ## Import a graph
    #graph = load_graph("tf_models/checkpoints/mnist/2018-02-16_02.18.04/frozen_model.pb", (1, 28, 28, 1), old_input="MnistNetwork_1/input:0")

    #for op in graph.get_operations():
    #    print(op.name)


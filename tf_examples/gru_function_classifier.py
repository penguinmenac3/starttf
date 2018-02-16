import math
import tensorflow as tf
import numpy as np

from utils.hyperparams import load_params
from utils.plot_losses import create_plot

from datasets.classification.function_generator import function_generator
from utils.generic_data_loader import load_data

from tf_models.gru_function_classifier import create_model, create_loss
from tf_models.model import train, export_graph, load_graph

GENERATE_DATA = False


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


def import_graph(checkpoint_path):
    hyper_params = load_params("tf_examples/gru_function_classifier.json")

    # Load Graph
    graph = load_graph(checkpoint_path + "/frozen_model.pb", (1, 100, 1), old_input="GruFunctionClassifier_1/Reshape:0")

    # Find relevant tensors
    input_tensor = graph.get_tensor_by_name('input_tensor:0')
    Hin = graph.get_tensor_by_name('GruFunctionClassifier_1/Hin:0')
    probs = graph.get_tensor_by_name('GruFunctionClassifier_1/probs:0')

    # Create/define inputs
    x = np.array([[[sin_fn(i, 0)] for i in range(100)]], dtype=np.float32)
    np_Hin = np.zeros([1, hyper_params.arch.hidden_layer_size * hyper_params.arch.hidden_layer_depth])

    with tf.Session(graph=graph) as sess:
        y_out = sess.run(probs, feed_dict={Hin: np_Hin, input_tensor: x})
        print(y_out)


def main():
    # Define "constants".
    hyper_params = load_params("tf_examples/gru_function_classifier.json")

    # Load training data
    print("Loading data")
    data_tmp_folder = "data/.records/gru_function_classifier"
    train_features, train_labels, validation_features, validation_labels = load_data(hyper_params, generate_data_fn,
                                                                                     data_tmp_folder)

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
    export_graph(checkpoint_path=checkpoint_path, output_nodes=["GruFunctionClassifier_1/probs"])
    print("Exported to: %s" % checkpoint_path)

    # To check if import works.
    #import_graph(checkpoint_path)


if __name__ == "__main__":
    main()

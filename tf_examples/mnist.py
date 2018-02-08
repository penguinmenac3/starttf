import math
import os
import tensorflow as tf

from datasets.classification.mnist import mnist
from datasets.tfrecords import write_tf_records, read_tf_records, PHASE_TRAIN, PHASE_VALIDATION
from tf_models.mnist import Mnist

GENERATE_DATA = False


def main():
    # Define "constants".
    hyper_params_filepath = "tf_examples/mnist.json"
    data_tmp_folder = "data/.records/mnist"
    base_dir = "data/mnist"
    validation_examples_number = 10000

    if GENERATE_DATA or not os.path.exists(data_tmp_folder):
        if not os.path.exists(data_tmp_folder):
            os.makedirs(data_tmp_folder)
        # Create training data.
        print("Generating data")
        train_data = mnist(base_dir=base_dir, phase=PHASE_TRAIN)
        validation_data = mnist(base_dir=base_dir, phase=PHASE_VALIDATION)

        # Write tf records
        print("Writing data")
        write_tf_records(data_tmp_folder, 4, 2, train_data, validation_data)

    # Create model.
    print("Creating Model")
    model = Mnist(hyper_params_filepath)

    # Load data with tf records.
    print("Loading data")
    train_features, train_labels = read_tf_records(data_tmp_folder, PHASE_TRAIN, model.hyper_params.train.batch_size)
    validation_features, validation_labels = read_tf_records(data_tmp_folder, PHASE_VALIDATION,
                                                             model.hyper_params.train.batch_size)

    # Limit used gpu memory.
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.75

    # train model.
    with tf.Session(config=config) as sess:
        print("Setup")
        model.setup(sess)

        print("Training")
        model.fit(train_features, train_labels, validation_examples_number, validation_features, validation_labels,
                  verbose=True)

        print("Exporting")
        model.export()


if __name__ == "__main__":
    main()

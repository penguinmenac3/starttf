import os

from starttf.utils.hyperparams import load_params
from starttf.tfrecords.autorecords import write_data

from opendatalake.classification.mnist import mnist

if __name__ == "__main__":
    # Load the hyper parameters.
    hyper_params = load_params("starttf/examples/mnist/hyper_params.json")

    # Get a generator and its parameters
    train_gen, train_gen_params = mnist(base_dir=hyper_params.problem.data_path, phase="train")
    validation_gen, validation_gen_params = mnist(base_dir=hyper_params.problem.data_path, phase="validation")

    # Create the paths where to write the records from the hyper parameter file.
    train_record_path = os.path.join(hyper_params.train.tf_records_path, "train")
    validation_record_path = os.path.join(hyper_params.train.tf_records_path, "validation")

    # Write the data
    write_data(hyper_params, train_record_path, train_gen, train_gen_params, 8)
    write_data(hyper_params, validation_record_path, validation_gen, validation_gen_params, 8)

import os
import math

from starttf.utils.hyperparams import load_params
from starttf.tfrecords.autorecords import write_data

from opendatalake.classification.function_generator import function_generator


def sin_fn(x, off):
    return math.sin(x / 50.0 + off)


def lin_fn(x, off):
    return x / 50.0 + off


if __name__ == "__main__":
    # Load the hyper parameters.
    hyper_params = load_params("starttf/examples/gru_function_classifier/hyper_params.json")

    # Get a generator and its parameters
    train_gen, train_gen_params = function_generator([sin_fn, lin_fn], 100, hyper_params.problem.training_examples)
    validation_gen, validation_gen_params = function_generator([sin_fn, lin_fn], 100, hyper_params.problem.validation_examples)

    # Create the paths where to write the records from the hyper parameter file.
    train_record_path = os.path.join(hyper_params.train.tf_records_path, "train")
    validation_record_path = os.path.join(hyper_params.train.tf_records_path, "validation")

    # Write the data
    write_data(hyper_params, train_record_path, train_gen, train_gen_params, 4)
    write_data(hyper_params, validation_record_path, validation_gen, validation_gen_params, 2)

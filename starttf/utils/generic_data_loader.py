import os
from opendatalake.tfrecords import write_tf_records, read_tf_records, PHASE_TRAIN, PHASE_VALIDATION


def load_data(hyper_params, generate_data_fn, data_tmp_folder, force_generate_data=False):
    if force_generate_data or not os.path.exists(data_tmp_folder):
        if not os.path.exists(data_tmp_folder):
            os.makedirs(data_tmp_folder)
        # Create training data.
        train_data, validation_data = generate_data_fn()

        # Write tf records
        write_tf_records(data_tmp_folder, 4, 2, train_data, validation_data)

    # Load data with tf records.
    train_features, train_labels = read_tf_records(data_tmp_folder, PHASE_TRAIN, hyper_params.train.batch_size)
    validation_features, validation_labels = read_tf_records(data_tmp_folder, PHASE_VALIDATION, hyper_params.train.validation_batch_size)

    return train_features, train_labels, validation_features, validation_labels

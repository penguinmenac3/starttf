import os
import numpy as np
from multiprocessing import Pool
import tensorflow as tf
from os import listdir
from os.path import isfile, join
import json

PHASE_TRAIN = "train"
PHASE_VALIDATION = "validation"


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _write_tf_record_pool_helper(args):
    hyper_params, data, num_threads, i, record_filename, preprocess_feature, preprocess_label, augment_data = args
    data_fn, data_params = data
    thread_name = "%s:thread_%d" % (record_filename, i)
    _write_tf_record(hyper_params, data_fn(data_params, num_threads, i), record_filename, preprocess_feature, preprocess_label, augment_data, thread_name=thread_name)


def _write_tf_record(hyper_params, data, record_filename, preprocess_feature=None, preprocess_label=None, augment_data=None, thread_name="thread"):
    writer = tf.python_io.TFRecordWriter(record_filename)

    samples_written = 0
    augmentation_steps = 1
    if "problem" in hyper_params.__dict__ and "augmentation" in hyper_params.__dict__:
        augmentation_steps = hyper_params.problem.augmentation.steps
    for orig_feature, orig_label in data:
        for i in range(augmentation_steps):
            feature = orig_feature
            label = orig_label
            if augment_data is not None:
                feature, label = augment_data(hyper_params, feature, label)
            if preprocess_feature is not None:
                feature = preprocess_feature(hyper_params, feature)
                if feature is None:
                    continue
            if preprocess_label is not None:
                label = preprocess_label(hyper_params, feature, label)
                if label is None:
                    continue

            feature_dict = {}

            for k in feature.keys():
                feature_dict['feature_' + k] = _bytes_feature(np.reshape(feature[k], (-1,)).tobytes())
            for k in label.keys():
                feature_dict['label_' + k] = _bytes_feature(np.reshape(label[k], (-1,)).tobytes())

            example = tf.train.Example(features=tf.train.Features(
                feature=feature_dict))
            writer.write(example.SerializeToString())
            samples_written += 1
            if samples_written % 1000 == 0:
                print("Samples written by %s: %d." % (thread_name, samples_written))
    print("Samples written by %s: %d." % (thread_name, samples_written))
    writer.close()


def _read_tf_record(record_filename, config):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(record_filename)

    feature_dict = {}
    for k in config.keys():
        if "feature_" in k or "label_" in k:
            feature_dict[k] = tf.FixedLenFeature([], tf.string)

    data = tf.parse_single_example(
        serialized_example,
        features=feature_dict)

    outputs = {}
    for k in feature_dict.keys():
        feature_shape = config[k]["shape"]
        feature_type = np.dtype(config[k]["dtype"])
        feature = tf.decode_raw(data[k], feature_type)
        feature_len = 1
        for x in list(feature_shape):
            feature_len *= x
        feature.set_shape((feature_len,))
        outputs[k] = feature

    return outputs


def _create_parser_fn(config, phase):
    def parser_fn(serialized_example):
        tensor_dict = {}
        for k in config.keys():
            if "feature_" in k or "label_" in k:
                tensor_dict[k] = tf.FixedLenFeature([], tf.string)

        data = tf.parse_single_example(
            serialized_example,
            features=tensor_dict)

        outputs = {}
        for k in tensor_dict.keys():
            tensor_shape = config[k]["shape"]
            tensor_type = np.dtype(config[k]["dtype"])
            tensor = tf.decode_raw(data[k], tensor_type)
            tensor_len = 1
            for x in list(tensor_shape):
                tensor_len *= x
            tensor.set_shape((tensor_len,))
            outputs[k] = tensor

        features = {}
        labels = {}
        for k in outputs.keys():
            shape = tuple(list(config[k]["shape"]))
            tensor = tf.reshape(outputs[k], shape, name="input/" + phase + "/" + k + "_reshape")
            if "feature_" in k:
                features["_".join(k.split("_")[1:])] = tensor
            if "label_" in k:
                labels["_".join(k.split("_")[1:])] = tensor

        return features, labels
    return parser_fn


def write_data(hyper_params,
               prefix,
               threadable_generator,
               params,
               num_threads,
               preprocess_feature=None,
               preprocess_label=None,
               augment_data=None):
    """
    Write a tf record containing a feature dict and a label dict.

    :param hyper_params: The hyper parameters required for writing {"problem": {"augmentation": {"steps": Int}}}
    :param prefix: The path prefix where to store your data. Recomennded is something like "data/.records/mnist/train" and "data/.records/mnist/validation"
    :param threadable_generator: A generator that supports a nicely threadable api.
        def gen(params, stride, offset, infinite)
        and returns a feature dict and a label dict containing a single example.
    :param params: The parameters for the threadable generator.
    :param num_threads: The number of threads. (Recommended: 4 for training and 2 for validation seems to works nice)
    :param preprocess_feature: A function that transforms the features from the raw dataset generator into something that your network could use.
        e.g. changing the encoding. (hyper_params, feature dict -> feature dict)
    :param preprocess_label: A function that transforms the labels from the raw dataset generator into something that your network could use.
        e.g. one hot encoding. (hyper_params, feature dict, label dict -> label dict)
    :param augment_data: A method that augments your data. eg random crop, scale, etc. (hyper_params, feature dict, label dict -> feature dict, label dict)
    :return:
    """
    prefix = prefix.replace("\\", "/")
    data_tmp_folder = "/".join(prefix.split("/")[:-1])
    if not os.path.exists(data_tmp_folder):
        os.makedirs(data_tmp_folder)

    args = [(hyper_params, (threadable_generator, params), num_threads, i, (prefix + "_%d.tfrecords") % i,
                   preprocess_feature, preprocess_label, augment_data) for i in range(num_threads)]

    # Retrieve a single sample
    data_gen = threadable_generator(params)
    sample_label = None
    sample_feature = None
    while sample_label is None or sample_feature is None:
        sample_feature, sample_label = next(data_gen)

        # Preprocess samples, so that shapes and dtypes are correct.
        if augment_data is not None:
            sample_feature, sample_label = augment_data(hyper_params, sample_feature, sample_label)
        if preprocess_feature is not None:
            sample_feature = preprocess_feature(hyper_params, sample_feature)
        if preprocess_label is not None:
            sample_label = preprocess_label(hyper_params, sample_feature, sample_label)

    config = {"num_threads": num_threads}
    for k in sample_feature.keys():
        config["feature_" + k] = {"shape": sample_feature[k].shape, "dtype": sample_feature[k].dtype.name}
    for k in sample_label.keys():
        config["label_" + k] = {"shape": sample_label[k].shape, "dtype": sample_label[k].dtype.name}

    with open(prefix + '_config.json', 'w') as outfile:
        json.dump(config, outfile)

    pool = Pool(processes=num_threads)
    pool.map(_write_tf_record_pool_helper, args)


def read_data_legacy(prefix, batch_size):
    """
    Loads a tf record as tensors you can use.
    :param prefix: The path prefix as defined in the write data method.
    :param batch_size: The batch size you want for the tensors.
    :return: A feature tensor dict and a label tensor dict.
    """
    prefix = prefix.replace("\\", "/")
    folder = "/".join(prefix.split("/")[:-1])
    phase = prefix.split("/")[-1]
    config = json.load(open(prefix + '_config.json'))
    num_threads = config["num_threads"]

    filenames = [folder + "/" + f for f in listdir(folder) if isfile(join(folder, f)) and phase in f and not "config.json" in f]

    # Create a tf object for the filename list and the readers.
    filename_queue = tf.train.string_input_producer(filenames)
    readers = [_read_tf_record(filename_queue, config) for _ in range(num_threads)]

    batch_dict = tf.train.shuffle_batch_join(
        readers,
        batch_size=batch_size,
        capacity=10 * batch_size,
        min_after_dequeue=5 * batch_size
    )

    # Add batch dimension to feature and label shape

    feature_batch = {}
    label_batch = {}
    for k in batch_dict.keys():
        shape = tuple([batch_size] + list(config[k]["shape"]))
        tensor = tf.reshape(batch_dict[k], shape, name="input/"+phase+"/" + k + "_reshape")
        if "feature_" in k:
            feature_batch["_".join(k.split("_")[1:])] = tensor
        if "label_" in k:
            label_batch["_".join(k.split("_")[1:])] = tensor

    return feature_batch, label_batch


def read_data(prefix, batch_size, augmentation=None):
    """
    Loads a dataset.

    :param prefix: The path prefix as defined in the write data method.
    :param batch_size: The batch size you want for the tensors.
    :param augmentation: An augmentation function.
    :return: A tensorflow.data.dataset object.
    """
    prefix = prefix.replace("\\", "/")
    folder = "/".join(prefix.split("/")[:-1])
    phase = prefix.split("/")[-1]
    config = json.load(open(prefix + '_config.json'))
    num_threads = config["num_threads"]

    filenames = [folder + "/" + f for f in listdir(folder) if isfile(join(folder, f)) and phase in f and not "config.json" in f]

    dataset = tf.data.TFRecordDataset(filenames=filenames, num_parallel_reads=num_threads)
    dataset = dataset.shuffle(buffer_size=10 * batch_size)
    dataset = dataset.repeat()
    dataset = dataset.map(map_func=_create_parser_fn(config, phase), num_parallel_calls=num_threads)
    if augmentation is not None:
        dataset = dataset.map(map_func=augmentation, num_parallel_calls=num_threads)
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.prefetch(buffer_size=1)

    return dataset


def create_input_fn(prefix, batch_size, augmentation=None):
    """
    Loads a dataset.

    :param prefix: The path prefix as defined in the write data method.
    :param batch_size: The batch size you want for the tensors.
    :param augmentation: An augmentation function.
    :return: An input function for a tf estimator.
    """
    def input_fn():
        return read_data(prefix, batch_size, augmentation)

    return input_fn


def create_legacy_input_fn(prefix, batch_size, augmentation=None):
    """
    Loads a dataset the old way.

    :param prefix: The path prefix as defined in the write data method.
    :param batch_size: The batch size you want for the tensors.
    :param augmentation: An augmentation function.
    :return: An input function for a tf estimator.
    """
    def input_fn():
        return read_data_legacy(prefix, batch_size)

    return input_fn


@DeprecationWarning
def auto_read_write_data(hyper_params, generate_data_fn, data_tmp_folder, force_generate_data=False, preprocess_feature=None, preprocess_label=None):
    """
    Deprecated: Do not use!
    """
    if force_generate_data or not os.path.exists(data_tmp_folder):
        if not os.path.exists(data_tmp_folder):
            os.makedirs(data_tmp_folder)
        # Create training data.
        train_data, validation_data = generate_data_fn()

        # Write tf records
        write_data(hyper_params, os.path.join(data_tmp_folder, PHASE_TRAIN), train_data[0], train_data[1], 4, preprocess_feature, preprocess_label)
        write_data(hyper_params, os.path.join(data_tmp_folder, PHASE_VALIDATION), validation_data[0], validation_data[1], 2, preprocess_feature, preprocess_label)

    # Load data with tf records.
    train_features, train_labels = read_data_legacy(os.path.join(data_tmp_folder, PHASE_TRAIN), hyper_params.train.batch_size)
    validation_features, validation_labels = read_data_legacy(os.path.join(data_tmp_folder, PHASE_VALIDATION), hyper_params.train.validation_batch_size)

    return train_features, train_labels, validation_features, validation_labels

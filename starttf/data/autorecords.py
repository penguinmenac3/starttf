# MIT License
# 
# Copyright (c) 2018 Michael Fuerst
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import numpy as np
from multiprocessing import Pool
import tensorflow as tf
from os import listdir
from os.path import isfile, join
import json
Sequence = tf.keras.utils.Sequence

PHASE_TRAIN = "train"
PHASE_VALIDATION = "validation"


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _write_tf_record_pool_helper(args):
    hyper_params, sequence, num_threads, i, record_filename = args
    thread_name = "%s:thread_%d" % (record_filename, i)
    _write_tf_record(hyper_params, sequence, num_threads, i, record_filename, thread_name=thread_name)


def _write_tf_record(hyper_params, sequence, num_threads, i, record_filename, thread_name="thread"):
    writer = tf.python_io.TFRecordWriter(record_filename)

    samples_written = 0
    augmentation_steps = hyper_params.problem.augmentation.get("steps", 1)

    for i in range(augmentation_steps):
        for idx in range(i, len(sequence), num_threads):
            feature_batch, label_batch = sequence[idx]
            batch_size = feature_batch.values()[0].shape[0]
            for batch_idx in range(batch_size):
                feature_dict = {}

                for k in feature_batch.keys():
                    feature_dict['feature_' + k] = _bytes_feature(np.reshape(feature_batch[k][batch_idx], (-1,)).tobytes())
                for k in label_batch.keys():
                    feature_dict['label_' + k] = _bytes_feature(np.reshape(label_batch[k][batch_idx], (-1,)).tobytes())

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


def _read_data_legacy(prefix, batch_size):
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


def _read_data(prefix, batch_size, augmentation=None):
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
    # Check if the version is too old for dataset api to work better than manually loading data.
    if tf.__version__.startswith("1.6") or tf.__version__.startswith("1.5") or tf.__version__.startswith("1.4") \
            or tf.__version__.startswith("1.3") or tf.__version__.startswith("1.2") \
            or tf.__version__.startswith("1.1") or tf.__version__.startswith("1.0"):
        def input_fn():
            with tf.variable_scope("input_pipeline"):
                return _read_data_legacy(prefix, batch_size)
        return input_fn
    else:
        def input_fn():
            with tf.variable_scope("input_pipeline"):
                return _read_data(prefix, batch_size, augmentation)
        return input_fn


def write_data(hyper_params,
               mode,
               sequence,
               num_threads):
    """
    Write a tf record containing a feature dict and a label dict.

    :param hyper_params: The hyper parameters required for writing {"problem": {"augmentation": {"steps": Int}}}
    :param mode: The mode specifies the purpose of the data. Typically it is either "train" or "validation".
    :param sequence: A tf.keras.utils.sequence.
    :param num_threads: The number of threads. (Recommended: 4 for training and 2 for validation seems to works nice)
    :return:
    """
    if not isinstance(sequence, Sequence) and not (callable(getattr(sequence, "__getitem__", None)) and callable(getattr(sequence, "__len__", None))):
        raise ValueError("sequence must be tf.keras.utils.Sequence or a subtype or implement __len__(self) and __getitem__(self, idx)")
    prefix = os.path.join(hyper_params.train.get("tf_records_path", "tfrecords"), mode)
    prefix = prefix.replace("\\", "/")
    data_tmp_folder = "/".join(prefix.split("/")[:-1])
    if not os.path.exists(data_tmp_folder):
        os.makedirs(data_tmp_folder)

    args = [(hyper_params, sequence, num_threads, i, (prefix + "_%d.tfrecords") % i) for i in range(num_threads)]

    # Retrieve a single batch
    sample_feature, sample_label = sequence[0]

    config = {"num_threads": num_threads}
    for k in sample_feature.keys():
        config["feature_" + k] = {"shape": sample_feature[k].shape[1:], "dtype": sample_feature[k].dtype.name}
    for k in sample_label.keys():
        config["label_" + k] = {"shape": sample_label[k].shape[1:], "dtype": sample_label[k].dtype.name}

    with open(prefix + '_config.json', 'w') as outfile:
        json.dump(config, outfile)

    pool = Pool(processes=num_threads)
    pool.map(_write_tf_record_pool_helper, args)

# MIT License
#
# Copyright (c) 2019 Michael Fuerst
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

from math import ceil
from numpy import array
import tensorflow as tf


class Sequence(tf.keras.utils.Sequence):
    """
    This simple sequence makes implementing keras sequences in the correct way far simpler.

    A subclass must implement a __num_samples() -> int method
    and a __get_sample(idx: int) -> (feature_dict, label_dict) method which returns a single sample.

    This class automatically then applies augmentation, then preparation and finally batches as specified in
    hyperparams.train.batch_size.
    """

    def __init__(self, hyperparams, phase, preprocess_fn=None, augmentation_fn=None):
        super().__init__()
        self.hyperparams = hyperparams
        self.preprocess_fn = preprocess_fn
        self.augmentation_fn = augmentation_fn
        self.phase = phase

    def num_samples(self):
        raise NotImplementedError(
            "A subclass must implement this function to find out how many training samples it has.")

    def get_sample(self, idx):
        raise NotImplementedError(
            "A subclass must implement this. Returns a tuple of (feature, label) representing a single training sample.")

    def __len__(self):
        return ceil(self.num_samples() / self.hyperparams.train.get("batch_size", 1))

    def __getitem__(self, index):
        features = []
        labels = []
        batch_size = self.hyperparams.train.get("batch_size", 1)
        for idx in range(index * batch_size, min((index + 1) * batch_size, self.num_samples())):
            feature, label = self.get_sample(idx)
            if self.augmentation_fn is not None:
                feature, label = self.augmentation_fn(self.hyperparams, feature, label)
            if self.preprocess_fn is not None:
                feature, label = self.preprocess_fn(self.hyperparams, feature, label)
            features.append(feature)
            labels.append(label)
        input_tensor_order = sorted(list(features[0].keys()))
        return {k: array([dic[k] for dic in features]) for k in input_tensor_order},\
               {k: array([dic[k] for dic in labels]) for k in labels[0]}

# MIT License
#
# Copyright (c) 2018-2019 Michael Fuerst
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

import tensorflow as tf
import starttf


class Metrics(tf.Module):
    def __init__(self, name="Metrics"):
        super().__init__(name=name)
        self.metrics = {}
        self.avg = {}
        self.values = {}

    def reset_avg(self):
        pass

    def summary(self):
        for k in self.values:
            tf.summary.scalar("metrics/{}".format(k), self.values[k],
                                step=starttf.train.samples_seen)

    def __call__(self, y_true, y_pred):
        if self.metrics is None:
            raise RuntimeError("You must specify self.losses before calling this module.")

        result = {}
        for k in self.metrics:
            if not isinstance(self.metrics[k], list):
                self.metrics[k] = [self.metrics[k]]

            for metric in self.metrics[k]:
                val = tf.reduce_mean(metric(y_true[k], y_pred[k]))
                name = metric.__name__
                if isinstance(metric, tf.Module):
                    name = metric.name
                result["{}/{}".format(k, name)] = val

        self.values = result
        self.avg = self.values
        return result

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

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import json
import os
import tensorflow as tf
try:
    from IPython.display import clear_output
    NO_IPYTHON = False
except ModuleNotFoundError:
    NO_IPYTHON = True


def create_keras_callbacks(hyperparams, log_dir, no_artifacts=False):
    callbacks = []
    if not no_artifacts:
        callbacks.append(TrainValTensorBoard(log_dir=log_dir, summary_steps=hyperparams.train.log_steps,
                                             histogram_freq=0, batch_size=hyperparams.train.batch_size, write_grads=False, write_images=False))
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(log_dir + "/model.hdf5",
                                                            monitor='val_loss', save_best_only=True, mode='auto'))
        callbacks.append(tf.keras.callbacks.CSVLogger(log_dir + "/results.csv", separator=','))
    callbacks.append(tf.keras.callbacks.TerminateOnNaN())
    return callbacks


class TrainValTensorBoard(tf.keras.callbacks.TensorBoard):
    def __init__(self, log_dir='./logs', summary_steps=None, **kwargs):
        # Make the original `TensorBoard` log to a subdirectory 'training'
        #training_log_dir = os.path.join(log_dir, 'train')
        training_log_dir = log_dir
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)

        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = os.path.join(log_dir, 'val')
        self.summary_steps = summary_steps
        self.counter = 0
        self.seen_samples = 0

    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tf.summary.create_file_writer(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def on_batch_end(self, batch, logs=None):
        self.counter += 1
        self.seen_samples += logs.get('size', 0)
        if self.counter % self.summary_steps == 0:
            with self.writer.as_default():
                for name, value in logs.items():
                    if name in ['batch', 'size']:
                        continue
                    tf.summary.scalar(name, value.item(),
                                      step=self.seen_samples)

    def on_epoch_end(self, epoch, logs=None):
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}

        with self.writer.as_default():
            for name, value in val_logs.items():
                tf.summary.scalar(name, value.item(),
                                  step=self.seen_samples)

        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(TrainValTensorBoard, self).on_epoch_end(self.seen_samples, logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()

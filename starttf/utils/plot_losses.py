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


def create_keras_callbacks(hyperparams, log_dir):
    callbacks = []
    callbacks.append(TrainValTensorBoard(log_dir=log_dir, summary_steps=hyperparams.train.summary_steps, histogram_freq=0, batch_size=hyperparams.train.batch_size, write_grads=False, write_images=False))
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(log_dir + "/model.hdf5", monitor='val_loss', save_best_only=True, mode='auto'))
    callbacks.append(tf.keras.callbacks.CSVLogger(log_dir + "/results.csv", separator=','))
    callbacks.append(tf.keras.callbacks.TerminateOnNaN())
    return callbacks


class TrainValTensorBoard(tf.keras.callbacks.TensorBoard):
    def __init__(self, log_dir='./logs', summary_steps=None, **kwargs):
        # Make the original `TensorBoard` log to a subdirectory 'training'
        training_log_dir = os.path.join(log_dir, 'training')
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)

        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = os.path.join(log_dir, 'validation')
        self.summary_steps = summary_steps
        self.counter = 0
        self.seen_samples = 0

    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def on_batch_end(self, batch, logs=None):
        self.counter += 1
        self.seen_samples += logs.get('size', 0)
        if self.counter % self.summary_steps == 0:
            for name, value in logs.items():
                if name in ['batch', 'size']:
                    continue
                summary = tf.Summary()
                summary_value = summary.value.add()
                summary_value.simple_value = value.item()
                summary_value.tag = name
                self.writer.add_summary(summary, self.seen_samples)
            self.writer.flush()

    def on_epoch_end(self, epoch, logs=None):
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, self.seen_samples)
        self.val_writer.flush()

        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(TrainValTensorBoard, self).on_epoch_end(self.seen_samples, logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()


class DefaultLossCallback(tf.train.SessionRunHook):
    def __init__(self, hyper_params, losses, checkpoint_dir, inline_plotting=False, report_storage={}, mode="train"):
        """
        A metric plotter and saver.

        Define your metrics in your loss by using the mode_str + "/*" pattern to add you metrics to the metrics dict
        and this callback will be capable auf automatically plotting them and saving them into a csv file.

        The metrics files can be found in your checkpoint folder.

        :param inline_plotting: This parameter is for jupyter notebook users. This will plot the loss not in a file but inside the notebook.
        """
        self.hyper_params = hyper_params
        self.losses = losses
        self.checkpoint_dir = checkpoint_dir
        self.report_storage = report_storage
        self.mode = mode
        self.inline_plotting = inline_plotting and not NO_IPYTHON
        if mode not in self.report_storage:
            if self.checkpoint_dir.endswith("/eval"):
                self.checkpoint_dir = self.checkpoint_dir[:-5]
            if os.path.exists(self.checkpoint_dir + "/images/record_storage.json"):
                with open(self.checkpoint_dir + "/images/record_storage.json", "r") as f:
                    self.report_storage = json.loads(f.read())
            self.plot_all()
        if mode not in self.report_storage:
            self.report_storage[mode] = {}

    def _compute_mean_per_step(self, mode, k):
        steps = []
        values = []
        n = []
        last_step = -1
        for step, val in zip(self.report_storage[mode]["step"], self.report_storage[mode][k]):
            if step != last_step:
                values.append(0)
                steps.append(step)
                n.append(0)
                last_step = step
            values[-1] += val
            n[-1] += 1

        for i in range(len(values)):
            values[i] = values[i] / float(n[i])

        return steps, values

    def plot_all(self):
        keys = list(self.report_storage.keys())
        if len(keys) == 0:
            return
        dummy_mode = keys[0]
        self.plot("loss")
        for k in sorted(list(self.report_storage[dummy_mode].keys())):
            if k == "step" or k == "loss":
                continue
            self.plot(k)

    def plot(self, k):
        data = []
        last_step = -1
        for mode in self.report_storage.keys():
            steps, values = self._compute_mean_per_step(mode, k)
            data.append((mode + "/" + k, steps, values))
        create_plot(k, self.checkpoint_dir, data, self.inline_plotting)

    def after_run(self, run_context, run_values):
        results = run_values.results
        if self.mode == "eval" or results["step"] % self.hyper_params.train.save_checkpoint_steps == 0:
            for k in results.keys():
                if k not in self.report_storage[self.mode]:
                    self.report_storage[self.mode][k] = []
                self.report_storage[self.mode][k].append(float(results[k]))

            if self.inline_plotting and self.mode != "eval":
                clear_output()

            if self.mode != "eval":
                print("{}: Step {}, Loss {}".format(self.mode, self.report_storage[self.mode]["step"][-1], self.report_storage[self.mode]["loss"][-1]))
                self.plot_all()
                with open(self.checkpoint_dir + "/images/record_storage.json", "w") as f:
                    f.write(json.dumps(self.report_storage, sort_keys=True, indent=4))

    def before_run(self, run_context):
        self.losses["step"] = tf.train.get_global_step()
        run_args = tf.train.SessionRunArgs(fetches=self.losses)
        return run_args

    def callback(self, i_step, metrics, reports, model_path):
        if self.inline_plotting:
            clear_output()
        # Interpret report matching to the report defined in reports by create_model
        print("Iter: %d, Train loss: %.4f, Test loss: %.4f" % (i_step, reports[0], reports[1]))
        self.iter_list.append(i_step)
        for i in range(len(reports)):
            if len(self.report_storage) <= i:
                self.report_storage.append([])
            self.report_storage[i].append(reports[i])

        # A map defining how to render the reports.
        plot_data = {}
        report_id = 0
        for i in range(len(metrics)):
            metric = metrics[i]
            for idx in range(len(metric.values())):
                label = list(metric.keys())[idx]
                report = self.report_storage[report_id]
                report_id += 1
                if label.split("/")[-1] not in plot_data:
                    plot_data[label.split("/")[-1]] = [label.split("/")[-1]]
                plot_data[label.split("/")[-1]].append((label, self.iter_list, report))

        for q in plot_data.values():
            create_plot(q[0], model_path, q[1:], self.inline_plotting)


def create_plot(title, model_path, data, inline_plotting=False):
    if not os.path.exists(model_path + "/images"):
        os.makedirs(model_path + "/images")

    plt.title(title)
    plt.xlabel("step")
    plt.ylabel(title)
    csv_dat = {}
    for date in data:
        label, x, y = date
        plt.plot(x, y, label=label)
        csv_dat[label + "/x"] = x
        csv_dat[label + "/y"] = y
    plt.legend()

    if inline_plotting:
        plt.show()
    else:
        plt.savefig(model_path + "/images/" + title.replace("/", ".") + ".png")
    plt.clf()


def f1_score(true, pred, f1_score_class, tresh=0.5):
    correct_positives = 0
    pred_positives = 0
    true_positives = 0

    for t, p in zip(true, pred):
        if t[f1_score_class] > 0.5 and p[f1_score_class] > tresh:
            correct_positives += 1
        if t[f1_score_class] > 0.5:
            true_positives += 1
        if p[f1_score_class] > tresh:
            pred_positives += 1

    if pred_positives > 0:
        precision = correct_positives / pred_positives
    else:
        precision = 0
    if true_positives > 0:
        recall = correct_positives / true_positives
    else:
        recall = 0
    if precision == 0 and recall == 0:
        return 0, 0, 0

    false_positive = pred_positives - correct_positives
    true_negative = len(true) - true_positives

    fpr = false_positive / true_negative
    tpr = correct_positives / true_positives
    return 2 * precision * recall / (precision + recall), tpr, fpr

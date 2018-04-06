import keras
import matplotlib.pyplot as plt
import numpy as np
import os


class DefaultLossCallback(object):
    def __init__(self):
        self.iter_list = []
        self.report_storage = []

    def callback(self, i_step, metrics, reports, model_path):
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
            create_plot(q[0], model_path, q[1:])


def create_plot(title, model_path, data):
    if not os.path.exists(model_path + "/images"):
        os.makedirs(model_path + "/images")

    plt.title(title)
    plt.xlabel("iter")
    plt.ylabel(title)
    for date in data:
        label, x, y = date
        plt.plot(x, y, label=label)
    plt.legend()
    plt.savefig(model_path + "/images/" + title + ".png")
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


class KerasPlotLosses(keras.callbacks.Callback):
    def __init__(self, loss_image_path, accuracy_image_path, f1_image_path=None, precision_recall_image_path=None, validation_data=None, f1_score_class=0):
        super().__init__()
        assert validation_data is not None

        self.loss_image_path = loss_image_path
        self.accuracy_image_path = accuracy_image_path
        self.f1_image_path = f1_image_path
        self.precision_recall_image_path = precision_recall_image_path
        self.validation_data = validation_data
        self.f1_score_class = f1_score_class
        self.i = 0
        self.x = []
        self.x_val = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []

        self.fig = plt.figure()

        self.logs = []

        self.val_f1s = []
        self.tpr = []
        self.fpr = []

    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.x_val = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []

        self.fig = plt.figure()

        self.logs = []

        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        self.i += 1
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))
        print("Epoch %04d: loss=%02.2f acc=%02.2f" % (self.i, self.losses[-1], self.acc[-1]))

        if self.i % 10 != 0:
            return

        data = next(self.validation_data)
        val_predict = np.asarray(self.model.predict(data[0]))
        val_targ = data[1]
        self.x_val.append(self.i)
        true_classes = np.argmax(data[1], axis=1)
        pred_classes = np.argmax(val_predict, axis=1)
        correct_classifications = 0
        for idx in range(len(true_classes)):
            if true_classes[idx] == pred_classes[idx]:
                correct_classifications += 1
        self.val_acc.append(correct_classifications / float(len(true_classes)))

        # TODO calculate validation loss
        self.val_losses.append(0)

        self.tpr = [1]
        self.fpr = [1]
        best_f1 = 0.0
        for idx in range(100):
            tresh = idx / 100.0
            _val_f1, tpr, fpr = f1_score(val_targ, val_predict, self.f1_score_class, tresh=tresh)
            self.tpr.append(tpr)
            self.fpr.append(fpr)
            best_f1 = max(best_f1, _val_f1)
        self.val_f1s.append(best_f1)
        self.tpr.append(0)
        self.fpr.append(0)

        print("Epoch %04d: loss=%02.2f acc=%02.2f val_loss=%02.2f val_acc=%02.2f f1=%02.2f" %
              (self.i, self.losses[-1], self.acc[-1], self.val_losses[-1], self.val_acc[-1], self.val_f1s[-1]))

        plt.xlabel("iter")
        plt.ylabel("loss")
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x_val, self.val_losses, label="val_loss")
        plt.legend()
        plt.savefig(self.loss_image_path)
        plt.clf()

        plt.xlabel("iter")
        plt.ylabel("acc")
        plt.plot(self.x, self.acc, label="acc")
        plt.plot(self.x_val, self.val_acc, label="val_acc")
        plt.legend()
        plt.savefig(self.accuracy_image_path)
        plt.clf()

        if self.f1_image_path is not None:
            plt.plot(self.x[0::10], self.val_f1s, label="f1_validation")
            plt.xlabel("iter")
            plt.ylabel("f1")
            plt.legend()
            plt.savefig(self.f1_image_path)
            plt.clf()

            plt.plot(self.fpr, self.tpr, label="roc_validation")
            plt.xlabel("fpr")
            plt.ylabel("tpr")
            plt.legend()
            plt.savefig(self.precision_recall_image_path)
            plt.clf()

        print("Plots updated.")

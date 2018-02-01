import keras
import matplotlib.pyplot as plt
import numpy as np


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


class PlotLosses(keras.callbacks.Callback):
    def __init__(self, loss_image_path, accuracy_image_path, f1_image_path=None, precision_recall_image_path=None, validation_data=None, f1_score_class=0):
        super().__init__()
        self.loss_image_path = loss_image_path
        self.accuracy_image_path = accuracy_image_path
        self.f1_image_path = f1_image_path
        self.precision_recall_image_path = precision_recall_image_path
        self.validation_data = validation_data
        self.f1_score_class = f1_score_class
        self.i = 0
        self.x = []
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
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))
        self.i += 1
        print("Iter %04d: loss=%02.2f acc=%02.2f val_loss=%02.2f val_acc=%02.2f" % (self.i, self.losses[-1], self.acc[-1], self.val_losses[-1], self.val_acc[-1]))
        if self.i % 10 != 0:
            return

        if self.i % 10 == 0:
            plt.xlabel("iter")
            plt.ylabel("loss")
            plt.plot(self.x, self.losses, label="loss")
            plt.plot(self.x, self.val_losses, label="val_loss")
            plt.legend()
            plt.savefig(self.loss_image_path)
            plt.clf()

            plt.xlabel("iter")
            plt.ylabel("acc")
            plt.plot(self.x, self.acc, label="acc")
            plt.plot(self.x, self.val_acc, label="val_acc")
            plt.legend()
            plt.savefig(self.accuracy_image_path)
            plt.clf()
            print("Loss & acc plots updated.")

        if self.f1_image_path is not None and self.validation_data is not None:
            val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
            val_targ = self.validation_data[1]
            self.tpr = [0]
            self.fpr = [0]
            best_f1 = 0.0
            for idx in range(100):
                tresh = idx / 100.0
                _val_f1, tpr, fpr = f1_score(val_targ, val_predict, self.f1_score_class, tresh=tresh)
                self.tpr.append(tpr)
                self.fpr.append(fpr)
                best_f1 = max(best_f1, _val_f1)
            self.val_f1s.append(best_f1)
            self.tpr.append(1)
            self.fpr.append(1)

            if self.i % 10 == 0:
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
                print("f1 & roc plots updated.")
        else:
            self.val_f1s.append(0)

        print("Iter %04d: loss=%02.2f acc=%02.2f val_loss=%02.2f val_acc=%02.2f f1=%02.2f" % (self.i, self.losses[-1], self.acc[-1], self.val_losses[-1], self.val_acc[-1], self.val_f1s[-1]))

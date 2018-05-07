import matplotlib.pyplot as plt
import numpy as np
import os
import json
from IPython.display import clear_output


class DefaultLossCallback(object):
    def __init__(self, inline_plotting=False):
        self.iter_list = []
        self.report_storage = []
        self.inline_plotting = inline_plotting

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
    plt.xlabel("iter")
    plt.ylabel(title)
    csv_dat = {}
    for date in data:
        label, x, y = date
        plt.plot(x, y, label=label)
        csv_dat[label + "/x"] = x
        csv_dat[label + "/y"] = y
    plt.legend()

    with open(model_path + "/images/" + title + ".csv", "w") as f:
        cols = csv_dat.keys()
        n_cols = len(cols)
        n_rows = max([len(csv_dat[cols[i]]) for i in range(n_cols)])

        csv = ",".join(cols) + "\n"
        for row in range(n_rows):
            line = []
            for col in range(n_cols):
                line.append("{}".format(csv_dat[cols[col]][row]))
            csv += ",".join(line) + "\n"

        f.write(csv)

    if inline_plotting:
        plt.show()
    else:
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

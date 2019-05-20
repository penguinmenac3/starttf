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

import pandas as pd

from starttf.data.simple_sequence import Sequence


class CSVSequence(Sequence):
    def __init__(self, hyperparams, phase, preprocess_fn=None, augmentation_fn=None):
        super().__init__(hyperparams, phase, preprocess_fn, augmentation_fn)
        file_name = hyperparams.problem.filename
        feature_name_list = hyperparams.problem.feature_name_list
        label_name_list = hyperparams.problem.label_name_list
        df = pd.read_csv(file_name)

        self.features = df.loc[:, feature_name_list].values
        self.labels = df.loc[:, label_name_list].values

    def num_samples(self):
        return len(self.features)

    def get_sample(self, idx):
        return {"feature": self.features[idx]}, {"label": self.labels[idx]}

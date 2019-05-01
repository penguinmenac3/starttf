from starttf.train import HyperParams

class Params(HyperParams):
    def __init__(self):
        super().__init__()
        self.train.experiment_name = "MNIST"

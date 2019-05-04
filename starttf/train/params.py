from hyperparams import HyperParams as OriginalParams


def __has_attribute(obj, name):
    return name in obj.__dict__ and obj.__dict__[name] is not None

def check_completness(params):
    # Check for training parameters
    assert __has_attribute(params, "train")
    assert __has_attribute(params.train, "experiment_name")
    assert __has_attribute(params.train, "checkpoint_path")
    assert __has_attribute(params.train, "batch_size")
    assert __has_attribute(params.train, "learning_rate")
    assert __has_attribute(params.train.learning_rate, "type")
    assert __has_attribute(params.train.learning_rate, "start_value")
    if params.train.learning_rate == "exponential":
        assert __has_attribute(params.train.learning_rate, "end_value")
    assert __has_attribute(params.train, "optimizer")
    assert __has_attribute(params.train.optimizer, "type")
    assert __has_attribute(params.train, "epochs")

    assert __has_attribute(params, "arch")
    assert __has_attribute(params.arch, "model")
    assert __has_attribute(params.arch, "loss")
    assert __has_attribute(params.arch, "eval")
    assert __has_attribute(params.arch, "prepare")

    assert __has_attribute(params, "problem")
    #assert __has_attribute(self.problem, "base_dir")


class HyperParams(OriginalParams):
    def __init__(self, d=None):
        self.train = OriginalParams()

        self.train.batch_size = 1
        self.train.experiment_name = None
        self.train.checkpoint_path = "checkpoints"
        self.train.epochs = 50
        self.train.learning_rate = OriginalParams()
        self.train.learning_rate.type = "const"
        self.train.learning_rate.start_value = 0.001
        self.train.learning_rate.end_value = 0.0001
        self.train.optimizer = OriginalParams()
        self.train.optimizer.type = "adam"

        self.arch = OriginalParams()
        self.arch.model = None
        self.arch.loss = None
        self.arch.eval = None
        self.arch.prepare = None

        self.problem = OriginalParams()
        self.problem.base_dir = None

        super().__init__(d=d)

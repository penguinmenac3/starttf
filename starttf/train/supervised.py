import sys
import tensorflow as tf
from hyperparams.hyperparams import load_params
import starttf
from starttf.utils.create_optimizer import create_keras_optimizer


PHASE_TRAIN = "train"
PHASE_VALIDATION = "validation"


#@tf.function
def __train(model, dataset, optimizer, loss_fn):
    i = 0
    N = len(dataset)
    for x, y in dataset:
        with tf.GradientTape() as tape:
            x["training"] = True
            prediction = model(**x)
            loss = loss_fn(y, prediction)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(gradients, model.trainable_variables)
        print("\rBatch {}/{} - {}".format(i+1, N, loss), end="")
        i += 1

#@tf.function
def __eval(model, dataset, eval_fn):
    total_loss = 0
    for x, y in dataset:
        prediction = model(x, training=False)
        total_loss += eval_fn(prediction, y)
    return total_loss / len(dataset)

def easy_train_and_evaluate(hyperparams, model=None, loss=None, evaluator=None, training_data=None, validation_data=None, optimizer=None, epochs=None, continue_training=False):
    starttf.hyperparams = hyperparams

    # Try to retrieve optional arguments from hyperparams if not specified
    if model is None:
        p = ".".join(hyperparams.arch.model.split(".")[:-1])
        n = hyperparams.arch.model.split(".")[-1]
        arch_model = __import__(p, fromlist=[n])
        model = arch_model.__dict__[n]()
    if loss is None and hyperparams.arch.get("loss", None) is not None:
        p = ".".join(hyperparams.arch.loss.split(".")[:-1])
        n = hyperparams.arch.loss.split(".")[-1]
        arch_loss = __import__(p, fromlist=[n])
        loss = arch_loss.__dict__[n]()
    if evaluator is None and hyperparams.arch.get("eval", None) is not None:
        p = ".".join(hyperparams.arch.eval.split(".")[:-1])
        n = hyperparams.arch.eval.split(".")[-1]
        arch_loss = __import__(p, fromlist=[n])
        evaluator = arch_loss.__dict__[n]()
    if training_data is None and hyperparams.arch.get("prepare", None) is not None:
        p = ".".join(hyperparams.arch.prepare.split(".")[:-1])
        n = hyperparams.arch.prepare.split(".")[-1]
        prepare = __import__(p, fromlist=[n])
        prepare = prepare.__dict__[n]
        training_data = prepare(hyperparams, PHASE_TRAIN)
        validation_data = prepare(hyperparams, PHASE_VALIDATION)
    if optimizer is None and hyperparams.train.get("optimizer", None) is not None:
        optimizer, lr_scheduler = create_keras_optimizer(hyperparams)
    if epochs is None:
        epochs = hyperparams.train.get("epochs", 1)

    # Check if all requirements could be retrieved.
    if model is None or loss is None or evaluator is None or training_data is None or validation_data is None or optimizer is None or epochs is None:
        raise RuntimeError("You must provide all arguments either directly or via hyperparams.")

    print("Epoch {}/{}".format(1, epochs))
    for i in range(epochs):
        __train(model, training_data, optimizer, loss)
        score = __eval(model, validation_data, evaluator)
        print("\rEpoch {}/{} - {}".format(i+1, epochs, score))


if __name__ == "__main__":
    if len(sys.argv) == 2 or len(sys.argv) == 3:
        continue_training = False
        idx = 1
        if sys.argv[idx] == "--continue":
            continue_training = True
            idx += 1
        hyperparams = load_params(sys.argv[1])
        easy_train_and_evaluate(hyperparams, continue_training=continue_training)
    else:
        print("Usage: python -m starttf.train.supervised [--continue] hyperparameters/myparams.json")

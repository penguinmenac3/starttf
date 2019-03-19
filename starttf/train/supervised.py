import sys
import tensorflow as tf
from hyperparams.hyperparams import load_params
from starttf.modules import module


PHASE_TRAIN = "train"
PHASE_VALIDATION = "validation"


@tf.function
def __train(model, dataset, optimizer, loss_fn):
    for x, y in dataset:
        with tf.GradientTape() as tape:
            prediction = model(x, training=True)
            loss = loss_fn(prediction, y)
    gradients = tape.gradients(loss, model.trainable_variables)
    optimizer.apply_gradients(gradients, model.trainable_variables)

@tf.function
def __eval(model, dataset, eval_fn):
    total_loss = 0
    for x, y in dataset:
        prediction = model(x, training=False)
        total_loss += eval_fn(prediction, y)
    return total_loss / len(dataset)

def easy_train_and_evaluate(hyperparams, model=None, loss_fn=None, eval_fn=None, training_data=None, validation_data=None, optimizer=None, epochs=None, continue_training=False):
    module.hyperparams = hyperparams

    # Try to retrieve optional arguments from hyperparams if not specified
    if model is None:
        arch_model = __import__(hyperparams.arch.model, fromlist=["Model"])
        model = arch_model.Model()
    if loss_fn is None and hyperparams.arch.get("loss", None) is not None:
        arch_loss = __import__(hyperparams.arch.loss, fromlist=["loss_fn"])
        loss_fn = arch_loss.loss_fn
    if eval_fn is None and hyperparams.arch.get("eval", None) is not None:
        arch_loss = __import__(hyperparams.arch.eval, fromlist=["eval_fn"])
        eval_fn = arch_loss.eval_fn
    if training_data is None and hyperparams.problem.get("prepare", None) is not None:
        prepare = __import__(hyperparams.problem.prepare, fromlist=["Sequence"])
        training_data = prepare.Sequence(hyperparams, PHASE_TRAIN)
        validation_data = prepare.Sequence(hyperparams, PHASE_VALIDATION)
    if optimizer is None and hyperparams.train.get("optimizer", None) is not None:
        optimizer = create_optimizer(hyperparams)
    if epochs is None:
        epochs = hyperparams.train.get("epochs", 1)

    # Check if all requirements could be retrieved.
    if model is None or loss_fn is None or eval_fn is None or training_data is None or validation_data is None or optimizer is None or epochs is None:
        raise RuntimeError("You must provide all arguments either directly or via hyperparams.")

    print("Epoch {}/{}".format(1, epochs))
    for i in range(epochs):
        __train(model, training_data, optimizer, loss_fn)
        score = __eval(model, validation_data, eval_fn)
        print("Epoch {}/{} - {}".format(i+1, epochs, score))


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

import starttf
import os
import numpy as np
from examples.fashion_mnist import FashionMnistModel, FashionMnistDataset, FashionMnistParams


def main():
    hyperparams = FashionMnistParams()
    hyperparams.train.batch_size = 1
    starttf.hyperparams = hyperparams
    dataset = FashionMnistDataset(hyperparams, "validation")
    features, labels = dataset[0]
    input_shapes_dict = {k: (features[k].shape[1], features[k].shape[2]) for k in features}
    input_dtypes_dict = {k: features[k].dtype for k in features}

    model = FashionMnistModel()
    latest_checkpoint = os.path.join(hyperparams.train.checkpoint_path, sorted(
        os.listdir(hyperparams.train.checkpoint_path))[-1])
    print("Loading Checkpoint: {}".format(latest_checkpoint))
    model.load_model(latest_checkpoint, input_shapes_dict, input_dtypes_dict)

    correct = 0
    wrong = 0
    for features, labels in dataset:
        output = model.predict(features)
        if np.argmax(output["class_id"]) == np.argmax(labels["class_id"]):
            correct += 1
        else:
            wrong += 1
    print("Accuracy: {}%".format(100 * correct / (correct + wrong)))


if __name__ == "__main__":
    main()

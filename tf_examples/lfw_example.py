import tensorflow as tf

from datasets.classification.named_folders import named_folders
from tf_models.lfw import LFWNetwork


def main():
    hyper_params_filepath = "examples/lfw.json"

    print("Load Dataset")
    roi = (200, 200)
    data = named_folders("data/lfw-deepfunneled", phase=None, crop_roi=roi, file_extension=".jpg")

    print("Create model")
    # TODO use a real model
    model = LFWNetwork(hyper_params_filepath)

    # Define how much gpu memory to allow for training.
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.75
    with tf.Session(config=config) as sess:
        print("Setup model")
        model.setup(sess)

        print("Fit model")
        model.fit(training_data=data, iters=50000)

        print("Export model")
        model.export()


if __name__ == "__main__":
    main()

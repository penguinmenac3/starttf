import tensorflow as tf


def get_default_config(gpu_memory_usage=0.75, allow_growth=False):
    """
    A helper to create sessions easily.
    :param gpu_memory_usage: How much of the gpu should be used for your project.
    :param allow_growth: If you want to have a fixed gpus size or if it should grow and use just as much as it needs.
    :return: A configuration you can pass to your session when creating it.
    """
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_usage
    config.gpu_options.allow_growth = allow_growth

    return config
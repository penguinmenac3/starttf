import tensorflow as tf

def get_default_config(gpu_memory_usage=0.75, allow_growth=False):
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_usage

    return config
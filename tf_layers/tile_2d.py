import tensorflow as tf

def tile_2d(input, k_x, k_y, name, reorder_required=True):
    size = input.get_shape().as_list()
    c, h, w = size[3], size[1], size[2]
    batch_size = size[0]

    # Check if tiling is possible and define output shape.
    assert c % (k_x * k_y) == 0

    tmp = input

    if reorder_required:
        output_channels = int(c / (k_x * k_y))
        channels = tf.unstack(tmp, axis=-1)
        reordered_channels = [None for _ in range(len(channels))]
        for o in range(output_channels):
            for i in range(k_x * k_y):
                target = o + i * output_channels
                source = o * (k_x * k_y) + i
                reordered_channels[target] = channels[source]
        tmp = tf.stack(reordered_channels, axis=-1)

    # Actual tilining
    with tf.variable_scope(name) as scope:
        tmp = tf.transpose(tmp, [0, 2, 1, 3])
        tmp = tf.reshape(tmp, (batch_size, w, int(h * k_y), int(c / (k_y))))
        tmp = tf.transpose(tmp, [0, 2, 1, 3])
        tmp = tf.reshape(tmp, (batch_size, int(h * k_y), int(w * k_x), int(c / (k_y * k_x))))
    
    return tmp

import time
import datetime
import tensorflow as tf
from tensorflow.python.tools.freeze_graph import freeze_graph
import json


def train(hyper_params, session, train_op, feed_dict):
    time_stamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H.%M.%S')

    # Init vars.
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    session.run(init_op)

    # Prepare training.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=session)
    saver = tf.train.Saver()

    # Merge all the summaries and write them out
    merged = tf.summary.merge_all()
    log_writer = tf.summary.FileWriter(hyper_params.train.checkpoint_path + "/" + time_stamp, session.graph)
    tf.global_variables_initializer().run()

    # Write hyper parameters to be able to track what config you had.
    with open(hyper_params.train.checkpoint_path + "/" + time_stamp + "/hyperparameters.json", "w") as json_file:
        json_file.write(json.dumps(hyper_params.to_dict(), indent=4, sort_keys=True))

    # Train
    print("Training Model: To reduce overhead no outputs are done. Use tensorboard to see your progress.")
    print("python -m tensorboard.main --logdir=tf_models/checkpoints")
    for i_step in range(hyper_params.train.iters):
        # Train step.
        if i_step % hyper_params.train.summary_iters != 0:
            session.run([train_op], feed_dict=feed_dict)
        else:  # Do validation and summary.
            _, summary = session.run([train_op, merged], feed_dict=feed_dict)
            log_writer.add_summary(summary, i_step)
            saver.save(session, hyper_params.train.checkpoint_path + "/" + time_stamp + "/chkpt",
                       global_step=i_step)
            print("Iter: %d" % i_step)

    saver.save(session, hyper_params.train.checkpoint_path + "/" + time_stamp + "/final_chkpt")
    coord.request_stop()
    coord.join(threads)

    return hyper_params.train.checkpoint_path + "/" + time_stamp


def export_graph(checkpoint_path, output_nodes):
    if not tf.gfile.Exists(checkpoint_path):
        raise AssertionError(
            "Export directory doesn't exists. Please specify an export "
            "directory: %s" % checkpoint_path)

    if not output_nodes:
        print("You need to supply the name of a node to --output_node_names.")
        return -1

        # We retrieve our checkpoint fullpath
    checkpoint = tf.train.get_checkpoint_state(checkpoint_path)
    input_checkpoint = checkpoint.model_checkpoint_path

    # We precise the file fullname of our freezed graph
    output_graph = checkpoint_path + "/frozen_model.pb"

    # We clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = True

    # We start a session using a temporary fresh Graph
    with tf.Session(graph=tf.Graph()) as sess:
        # We import the meta graph in the current default Graph
        saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

        # We restore the weights
        saver.restore(sess, input_checkpoint)

        # We use a built-in TF helper to export variables to constants
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,  # The session is used to retrieve the weights
            tf.get_default_graph().as_graph_def(),  # The graph_def is used to retrieve the nodes
            output_nodes  # The output node names are used to select the useful nodes
        )

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))

    return output_graph_def


def load_graph(frozen_graph_filename, input_shape, namespace_prefix="", old_input=None):
    # Load graph def from protobuff and import the definition
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    if old_input is None:
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name=namespace_prefix)
    else:
        with tf.Graph().as_default() as graph:
            tf_new_input = tf.placeholder(tf.float32, shape=input_shape, name="input_tensor")
            tf.import_graph_def(graph_def, input_map={old_input: tf_new_input}, name=namespace_prefix)
    return graph
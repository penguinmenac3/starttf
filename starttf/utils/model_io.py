# MIT License
# 
# Copyright (c) 2018 Michael Fuerst
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import tensorflow as tf


def export_graph(checkpoint_path, output_nodes):
    """
    Export a graph stored in a checkpoint as a *.pb file.
    :param checkpoint_path: The checkpoint path which should be frozen.
    :param output_nodes: The output nodes you care about as a list of strings (their names).
    :return:
    """
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
    """
    Loads a frozen graph from a *.pb file.
    :param frozen_graph_filename: The file which graph to load.
    :param input_shape: The shape of the new input you want.
    :param namespace_prefix: A namespace for your graph to live in. This is useful when having multiple models.
    :param old_input: The name of the old input tensor you want to replace.
    :return: The graph that can now be passed to a session when creating it.
    """
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

import sys
import numpy as np
import tensorflow as tf

from starttf.utils.hyperparams import load_params
from starttf.models.utils import load_graph
from starttf.examples.gru_function_classifier.prepare_training import sin_fn


def deploy_test(checkpoint_path):
    # Load Graph
    graph = load_graph(checkpoint_path + "/frozen_model.pb", (1, 100, 1), old_input="GruFunctionClassifier_2/Reshape:0")

    # Find relevant tensors
    input_tensor = graph.get_tensor_by_name('input_tensor:0')
    probs = graph.get_tensor_by_name('GruFunctionClassifier_2/probs:0')

    # Create/define inputs
    x = np.array([[[sin_fn(i, 0)] for i in range(100)]], dtype=np.float32)

    with tf.Session(graph=graph) as sess:
        y_out = sess.run(probs, feed_dict={input_tensor: x})
        print(y_out)


if __name__ == "__main__":
    # To check if import works.
    deploy_test(sys.argv[1])

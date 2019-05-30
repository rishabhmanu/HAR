import argparse
import tensorflow as tf

# Sample input of size 200x3
SAMPLE_INPUT = [[3, 5, 7],
[4, 5, 1],
[1, 1, 1],
[1, 1, 1],
[3, 5, 7],
[4, 5, 1],
[1, 1, 1],
[1, 1, 1],
[3, 5, 7],
[4, 5, 1],
[1, 1, 1],
[1, 1, 1],
[3, 5, 7],
[4, 5, 1],
[1, 1, 1],
[1, 1, 1],
[3, 5, 7],
[4, 5, 1],
[1, 1, 1],
[1, 1, 1],
[3, 5, 7],
[4, 5, 1],
[1, 1, 1],
[1, 1, 1],
[3, 5, 7],
[4, 5, 1],
[1, 1, 1],
[1, 1, 1],
[3, 5, 7],
[4, 5, 1],
[1, 1, 1],
[1, 1, 1],
[3, 5, 7],
[4, 5, 1],
[1, 1, 1],
[1, 1, 1],
[3, 5, 7],
[4, 5, 1],
[1, 1, 1],
[1, 1, 1],
[3, 5, 7],
[4, 5, 1],
[1, 1, 1],
[1, 1, 1],
[3, 5, 7],
[4, 5, 1],
[1, 1, 1],
[1, 1, 1],
[3, 5, 7],
[4, 5, 1],
[1, 1, 1],
[1, 1, 1],
[3, 5, 7],
[4, 5, 1],
[1, 1, 1],
[1, 1, 1],
[3, 5, 7],
[4, 5, 1],
[1, 1, 1],
[1, 1, 1],
[3, 5, 7],
[4, 5, 1],
[1, 1, 1],
[1, 1, 1],
[3, 5, 7],
[4, 5, 1],
[1, 1, 1],
[1, 1, 1],
[3, 5, 7],
[4, 5, 1],
[1, 1, 1],
[1, 1, 1],
[3, 5, 7],
[4, 5, 1],
[1, 1, 1],
[1, 1, 1],
[3, 5, 7],
[4, 5, 1],
[1, 1, 1],
[1, 1, 1],
[3, 5, 7],
[4, 5, 1],
[1, 1, 1],
[1, 1, 1],
[3, 5, 7],
[4, 5, 1],
[1, 1, 1],
[1, 1, 1],
[3, 5, 7],
[4, 5, 1],
[1, 1, 1],
[1, 1, 1],
[3, 5, 7],
[4, 5, 1],
[1, 1, 1],
[1, 1, 1],
[3, 5, 7],
[4, 5, 1],
[1, 1, 1],
[1, 1, 1],
[3, 5, 7],
[4, 5, 1],
[1, 1, 1],
[1, 1, 1],
[3, 5, 7],
[4, 5, 1],
[1, 1, 1],
[1, 1, 1],
[3, 5, 7],
[4, 5, 1],
[1, 1, 1],
[1, 1, 1],
[3, 5, 7],
[4, 5, 1],
[1, 1, 1],
[1, 1, 1],
[3, 5, 7],
[4, 5, 1],
[1, 1, 1],
[1, 1, 1],
[3, 5, 7],
[4, 5, 1],
[1, 1, 1],
[1, 1, 1],
[3, 5, 7],
[4, 5, 1],
[1, 1, 1],
[1, 1, 1],
[3, 5, 7],
[4, 5, 1],
[1, 1, 1],
[1, 1, 1],
[3, 5, 7],
[4, 5, 1],
[1, 1, 1],
[1, 1, 1],
[3, 5, 7],
[4, 5, 1],
[1, 1, 1],
[1, 1, 1],
[3, 5, 7],
[4, 5, 1],
[1, 1, 1],
[1, 1, 1],
[3, 5, 7],
[4, 5, 1],
[1, 1, 1],
[1, 1, 1],
[3, 5, 7],
[4, 5, 1],
[1, 1, 1],
[1, 1, 1],
[3, 5, 7],
[4, 5, 1],
[1, 1, 1],
[1, 1, 1],
[3, 5, 7],
[4, 5, 1],
[1, 1, 1],
[1, 1, 1],
[4, 5, 1],
[1, 1, 1],
[1, 1, 1],
[3, 5, 7],
[4, 5, 1],
[1, 1, 1],
[1, 1, 1],
[3, 5, 7],
[4, 5, 1],
[1, 1, 1],
[1, 1, 1],
[4, 5, 1],
[1, 1, 1],
[1, 1, 1],
[3, 5, 7],
[4, 5, 1],
[1, 1, 1],
[1, 1, 1],
[3, 5, 7],
[4, 5, 1],
[1, 1, 1],
[1, 1, 1],
[4, 5, 1],
[1, 1, 1],
[1, 1, 1],
[3, 5, 7],
[4, 5, 1],
[1, 1, 1],
[1, 1, 1],
[3, 5, 7],
[4, 5, 1],
[1, 1, 1],
[1, 1, 1],
[1, 1, 1],
[3, 5, 7],
[4, 5, 1],
[1, 1, 1],
[1, 1, 1],
[1, 1, 1],
[1, 1, 1]]

CLASS_LABELS = [
  "Downstairs",
  "Jogging",
  "Sitting",
  "Standing",
  "Upstairs",
  "Walking"
]

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph

if __name__ == '__main__':
    # Let's allow the user to pass the filename as an argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen_model_filename", default="frozen_model.pb", type=str, help="Frozen model file to import")
    args = parser.parse_args()

    # We use our "load_graph" function
    graph = load_graph(args.frozen_model_filename)

    # We can verify that we can access the list of operations in the graph
    # for op in graph.get_operations():
    #     print(op.name)

    # We access the input and output nodes
    x = graph.get_tensor_by_name('prefix/inputs:0') # 1 x 200 x 3
    y = graph.get_tensor_by_name('prefix/y_:0') # 1 x 6

    # We launch a Session
    with tf.Session(graph=graph) as sess:
        # Note: we don't need to initialize/restore anything
        y_out = sess.run(y, feed_dict={
            x: [SAMPLE_INPUT]
        })

        class_index = sess.run(tf.argmax(y_out[0], 0))

        print(y_out[0])
        print(CLASS_LABELS[class_index])

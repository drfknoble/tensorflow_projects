'''getting_started'''

# pylint: disable=E0401
# pylint: disable=C0103

# Import tensorflow to get access to tensorflow library
import tensorflow as tf 

# Create placeholders as empty variables, which you send data to when evaluating the graph
A = tf.placeholder(tf.float32, None, 'A')
B = tf.placeholder(tf.float32, None, 'B')

# Creates a simple graph of a single node, which adds two tensors
C = A + B

with tf.Session() as s:

    # Evaluate the graph, providing A = 1.0 and B = 2.0
    ans = s.run(C, {A: 1.0, B: 2.0})

    # Display answer on the console
    print(ans)

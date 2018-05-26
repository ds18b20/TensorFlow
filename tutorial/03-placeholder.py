import tensorflow as tf
import numpy as np

# basic
input_1 = tf.placeholder(tf.float32, name='input_1')
input_2 = tf.placeholder(tf.float32, name='input_2')

output = tf.multiply(input_1, input_2, name='multiply')

with tf.Session() as sess:
    # show in Tensorboard
    writer = tf.summary.FileWriter("./graphs", sess.graph)
    print(sess.run(output, feed_dict={input_1: [3.], input_2: [2.]}))

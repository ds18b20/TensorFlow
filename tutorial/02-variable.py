import tensorflow as tf
import numpy as np

# basic
state = tf.Variable(0, name='state')
# print(state.name)
one = tf.constant(1, name='one_con')
new_value = tf.add(state, one, name='new_val')
update = tf.assign(state, new_value, name='assign')

init = tf.global_variables_initializer()

with tf.Session() as sess:
    # show in Tensorboard
    writer = tf.summary.FileWriter("./graphs", sess.graph)
    sess.run(init)
    for _ in range(3):
        print(sess.run(update))

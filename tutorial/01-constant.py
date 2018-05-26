import tensorflow as tf
import numpy as np

# basic
a = tf.constant(2, name='ax')
b = tf.constant(3, name='bx')
x = tf.add(a, b, name='add')
y = tf.constant(np.arange(12).reshape(3, 4), shape=(2, 6))

# zeros/zeros_like
z = tf.zeros([2, 3], dtype=tf.int32, name='z')
zl = tf.zeros_like(np.arange(6).reshape(2, 3), dtype=tf.int32, name='zl')
zz = tf.multiply(z, zl, name='multiply')

# ones/ones_like
o = tf.ones([2, 3], dtype=tf.int32, name='o')
ol = tf.ones_like(np.arange(6).reshape(2, 3), dtype=tf.int32, name='ol')

# fill array with a value
f = tf.fill(dims=[2, 3], value=6, name='f')

# sequences
# start must be float
# [s, e] both s and e are included!!!
ls = tf.linspace(start=1.0, stop=10, num=10, name='ls')
rg = tf.range(3, name='rg') # [0, 1, 2] same to Numpy

# random
rn = tf.random_normal(shape=[2, 3], mean=1.0, stddev=1.0, dtype=tf.float32, name='rn')

with tf.Session() as sess:
    # show in Tensorboard
    writer = tf.summary.FileWriter("./graphs", sess.graph)
    print(sess.run(x))
    print(sess.run(y))
    # run multiple ops, ops should be put in []
    # x, y = sess.run(x, y) # error!
    x, y = sess.run([x, y])
    print(x, y)

    print(sess.run(z))
    print(sess.run(zl))

    print(sess.run(o))
    print(sess.run(ol))

    print(sess.run(f))

    print(sess.run(ls))
    print(sess.run(rg))

    print(sess.run(rn))

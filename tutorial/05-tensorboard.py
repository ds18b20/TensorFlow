import tensorflow as tf
import numpy as np


def add_layer(inputs, in_size, out_size, activation_function=None):
    with tf.name_scope('layer'):
        with tf.name_scope('Weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name = 'W')
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
        with tf.name_scope('W_m_p'):
            W_m_p = tf.matmul(inputs, Weights) + biases
        if activation_function is None:
            outputs = W_m_p
        else:
            outputs = activation_function(W_m_p)

        return outputs

# real data for placeholder
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

# placeholder for inputs data
with tf.name_scope('inputs'):
    xs = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='x_inputs')
    ys = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='y_inputs')

# add hidden layer
layer_1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
# add prediction layer
prediction = add_layer(layer_1, 10, 1, activation_function=None)
# loss function
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
# training
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)
# initialize
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    write = tf.summary.FileWriter('./graphs/graph', sess.graph)

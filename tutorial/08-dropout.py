import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import matplotlib.pyplot as plt


# load data
digits = load_digits()
X = digits.data
y = digits.target
y = LabelBinarizer().fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


def add_layer(inputs, in_size, out_size, layer_name, activation_function=None):
    with tf.name_scope('layer'):
        with tf.name_scope('Weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
            tf.summary.histogram(layer_name + '/Weights', Weights)
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
            tf.summary.histogram(layer_name + '/biases', biases)
        with tf.name_scope('W_m_p'):
            W_m_p = tf.matmul(inputs, Weights) + biases
        with tf.name_scope('Dropout'):
            W_m_p = tf.nn.dropout(W_m_p, keep_prob)
        if activation_function is None:
            outputs = W_m_p
        else:
            outputs = activation_function(W_m_p)
            tf.summary.histogram(layer_name + '/outputs', outputs)
        return outputs


# real data for placeholder

# placeholder for inputs data
with tf.name_scope('inputs'):
    keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob4dropout')
    xs = tf.placeholder(dtype=tf.float32, shape=[None, 64], name='x_inputs')
    ys = tf.placeholder(dtype=tf.float32, shape=[None, 10], name='y_inputs')

# add hidden layer
layer_1 = add_layer(xs, 64, 100, 'hidden_layer_1', activation_function=tf.nn.tanh)
# add prediction layer
prediction = add_layer(layer_1, 100, 10, 'prediction_layer', activation_function=tf.nn.softmax)

# loss function
with tf.name_scope('loss'):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))
    tf.summary.scalar('loss', cross_entropy)
# training
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(learning_rate=0.005).minimize(cross_entropy)

# initialize
init = tf.global_variables_initializer()

# Session
sess = tf.Session()
sess.run(init)
summaries = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('./graphs/dropout/train', sess.graph)
test_writer = tf.summary.FileWriter('./graphs/dropout/test', sess.graph)

# start training
for step in range(1000):
    sess.run(train_step, feed_dict={xs: X_train, ys: y_train, keep_prob: 0.6})
    if step % 50 == 0:
        # print(compute_accuracy(mnist.test.images, mnist.test.labels))
        train_result = sess.run(summaries, feed_dict={xs: X_train, ys: y_train, keep_prob: 1})
        test_result = sess.run(summaries, feed_dict={xs: X_test, ys: y_test, keep_prob: 1})
        train_writer.add_summary(train_result, global_step=step)
        test_writer.add_summary(test_result, global_step=step)
print('OK')

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


def compute_accuracy(xs_val, ys_val):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: xs_val, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre, axis=1), tf.argmax(ys_val, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: xs_val, ys: ys_val, keep_prob: 1})

    return result


# define weight
def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)


# define weight
def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


# define conv layer
def conv2d(x, W):
    # strides=[1,x_movement,y_movement,1]
    # padding: SAME size with input
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# define pooling layer
def max_pool_2x2(x):
    # ksize=[1, width, height, 1]
    # strides=[1,x_movement,y_movement,1]
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# placeholder for inputs data
with tf.name_scope('inputs'):
    xs = tf.placeholder(dtype=tf.float32, shape=[None, 28*28], name='x_inputs')
    ys = tf.placeholder(dtype=tf.float32, shape=[None, 10], name='y_inputs')

    # sample count * width * height * channel
    x_image = tf.reshape(xs, [-1, 28, 28, 1])
    image_summary = tf.summary.image("images_summary", x_image, max_outputs=4)

# convolutional layer 1
with tf.name_scope('conv_layer_1'):
    # kernal width: 5
    # kernal height: 5
    # kernal channel: 1
    # kernal count: 32
    with tf.name_scope('weights'):
        W_conv1 = weight_variable([5, 5, 1, 32], 'W_for_Conv_Layer_1')
        # Transpose Tensor dimensions from [5, 5, 1, 32] to [32, 5, 5, 1]
        tf.summary.image("kernels_1", tf.transpose(W_conv1, perm=[3, 0, 1, 2]), max_outputs=4)

    with tf.name_scope('bias'):
        b_conv1 = bias_variable([32], 'bias_for_Conv_Layer_1')

    with tf.name_scope('activations'):
        h_conv1 = conv2d(x_image, W_conv1) + b_conv1  # output size -1*28*28*32
        h_conv1 = tf.nn.relu(h_conv1)
        # Transpose Tensor dimensions from [-1, 28, 28, 32] to [-1, 32, 28, 28]
        # Merge dimension [-1] and [32], and show image
        tf.summary.image('activations_1', tf.reshape(tf.transpose(h_conv1, perm=[0, 3, 1, 2]), [-1, 28, 28, 1]),
                         max_outputs=4)

# max pool layer 1
with tf.name_scope('max_pool_layer_1'):
    h_pool1 = max_pool_2x2(h_conv1)  # output size -1*14*14*32
    # Transpose Tensor dimensions from [-1, 14, 14, 32] to [-1, 32, 14, 14]
    # Merge dimension [-1] and [32], and show image
    tf.summary.image('max_pool_1', tf.reshape(tf.transpose(h_conv1, perm=[0, 3, 1, 2]), [-1, 14, 14, 1]),
                     max_outputs=4)

# convolutional layer 2
with tf.name_scope('conv_layer_2'):
    # kernal width: 5
    # kernal height: 5
    # kernal channel: 32
    # kernal count: 64
    with tf.name_scope('weights'):
        W_conv2 = weight_variable([5, 5, 32, 64], 'W_for_Conv_Layer_2')
        # Transpose Tensor dimensions from [5, 5, 32, 64] to [32*64, 5, 5, 1]
        tf.summary.image('kernels_2', tf.reshape(tf.transpose(W_conv2, perm=[2, 3, 0, 1]), [-1, 5, 5, 1]),
                         max_outputs=4)

    with tf.name_scope('bias'):
        b_conv2 = bias_variable([64], 'bias_2')

    with tf.name_scope('activations'):
        h_conv2 = conv2d(h_pool1, W_conv2) + b_conv2  # output size -1*14*14*64
        h_conv2 = tf.nn.relu(h_conv2)
        # Transpose Tensor dimensions from [-1, 14, 14, 64] to [-1, 64, 14, 14]
        # Merge dimension [-1] and [64], and show image
        tf.summary.image('activations_2', tf.reshape(tf.transpose(h_conv2, perm=[0, 3, 1, 2]), [-1, 14, 14, 1]),
                         max_outputs=4)

# max pool layer 2
with tf.name_scope('max_pool_layer_2'):
    h_pool2 = max_pool_2x2(h_conv2)  # output size -1*7*7*64
    # Transpose Tensor dimensions from [-1, 7, 7, 64] to [-1, 64, 7, 7]
    # Merge dimension [-1] and [64], and show image
    tf.summary.image('max_pool_2', tf.reshape(tf.transpose(h_conv1, perm=[0, 3, 1, 2]), [-1, 7, 7, 1]),
                     max_outputs=4)

# fully connected layer 1
with tf.name_scope('fully_connected_layer_1'):
    W_fc1 = weight_variable([7*7*64, 1024], 'W_for_Fully_Connected_layer_1')
    b_fc1 = bias_variable([1024], 'bias_for_Fully_Connected_layer_1')
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])  # shape: [7, 7, 64]-->[-1, 7*7*64]
    h_fc1 = tf.matmul(h_pool2_flat, W_fc1) + b_fc1
    h_fc1 = tf.nn.relu(h_fc1)
    # Also add histograms to TensorBoard
    W_fc1_hist = tf.summary.histogram("W_fc1", W_fc1)
    b_fc1_hist = tf.summary.histogram("b_fc1", b_fc1)

    keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob4dropout')
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# fully connected layer 2
with tf.name_scope('fully_connected_layer_2'):
    W_fc2 = weight_variable([1024, 10], 'W_for_Fully_Connected_layer_2')
    b_fc2 = bias_variable([10], 'bias_for_Fully_Connected_layer_2')

    prediction = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    prediction = tf.nn.softmax(prediction)

    # Also add histograms to TensorBoard
    W_fc2_hist = tf.summary.histogram("W_fc2", W_fc2)
    b_fc2_hist = tf.summary.histogram("b_fc2", b_fc2)

# loss function
with tf.name_scope('loss'):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))
    tf.summary.scalar('loss', cross_entropy)

# training
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cross_entropy)

# initialize
init = tf.global_variables_initializer()

sess = tf.Session()

sess.run(init)
summaries = tf.summary.merge_all()
writer = tf.summary.FileWriter('./graphs/cnn_', sess.graph)

# start training
for step in range(1000):
    xs_batch, ys_batch = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: xs_batch, ys: ys_batch, keep_prob: 0.6})
    if step % 50 == 0:
        print(compute_accuracy(mnist.test.images[:100], mnist.test.labels[:100]))
        summary_result = sess.run(summaries, feed_dict={xs: xs_batch, ys: ys_batch, keep_prob: 1})
        writer.add_summary(summary_result, global_step=step)
        writer.flush()
print('OK')

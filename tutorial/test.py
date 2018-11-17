import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def add_layer(input_data, in_size, out_size, activator=None):
    weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]))
    Affine = tf.add(tf.matmul(input_data, weights), biases)
    if activator is None:
        output_data = Affine
    else:
        output_data = activator(Affine)

    return output_data


# sample data
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

xs = tf.placeholder(dtype=tf.float32, shape=[None, 1])
ys = tf.placeholder(dtype=tf.float32, shape=[None, 1])

layer_1 = add_layer(input_data=xs, in_size=1, out_size=10, activator=tf.nn.relu)
prediction = add_layer(input_data=layer_1, in_size=10, out_size=1, activator=None)

# loss function
loss = tf.reduce_mean(tf.square(ys - prediction))  # no reduction_indices selected
# loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=1))

# training
# train_step = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)  # Adam
train_step = tf.train.GradientDescentOptimizer(learning_rate=1e-3).minimize(loss)  # Gradient Descent
# initialize
init = tf.global_variables_initializer()

loss_lst =[]
N = 2000
epoch = 100
with tf.Session() as sess:
    sess.run(init)
    for i in range(N):
        sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
        ls = sess.run(loss, feed_dict={xs: x_data, ys: y_data})
        loss_lst.append(ls)
        if i % epoch == 0:
            print(ls)

    colors = np.random.rand(N)
    plt.scatter(range(N), loss_lst, c='grey',marker='X')
    plt.xlabel('training step')
    plt.ylabel('loss')
    plt.show()
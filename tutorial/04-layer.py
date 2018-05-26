import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    # W_m_p = tf.multiply(inputs, Weights) + biases
    W_m_p = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = W_m_p
    else:
        outputs = activation_function(W_m_p)

    return outputs


x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

xs = tf.placeholder(dtype=tf.float32, shape=[None, 1])
ys = tf.placeholder(dtype=tf.float32, shape=[None, 1])

layer_1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
prediction = add_layer(layer_1, 10, 1, activation_function=None)
# loss function
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
# training
train_step = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)
# initialize
init = tf.global_variables_initializer()

loss_lst =[]
N = 100
fig = plt.figure()
ax_1 = fig.add_subplot(1, 2, 1)
ax_2 = fig.add_subplot(1, 2, 2)
ax_2.scatter(x_data, y_data)

with tf.Session() as sess:
    sess.run(init)
    for i in range(N):
        sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
        ls = sess.run(loss, feed_dict={xs: x_data, ys: y_data})
        loss_lst.append(ls)
        colors = np.random.rand(N)
        if i % 10 == 0:
            # print(ls)
            try:
                ax_2.lines.remove(lines[0])
            except Exception:
                pass
            prediction_val = sess.run(prediction, feed_dict={xs: x_data})
            # ax_2.scatter(x_data, prediction_val, c='black', marker='X')
            lines = ax_2.plot(x_data, prediction_val, c='black', marker='X')
            plt.pause(0.25)
    ax_1.scatter(range(N), loss_lst, c='grey', marker='X')

    plt.xlabel('training step')
    plt.ylabel('loss')
    plt.show()

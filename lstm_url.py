
import tensorflow as tf
from tensorflow.contrib import rnn
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# print(mnist.test.images[0], mnist.test.labels[0])


config = tf.ConfigProto()
sess = tf.Session(config=config)

lr = 1e-3
batch_size = tf.placeholder(tf.int32, [])
input_size = 20
timestep_size = 10
hidden_size = 128
layer_num = 2
class_num = 2

_X = tf.placeholder(tf.float32, [None, 200])
y = tf.placeholder(tf.float32, [None, class_num])
keep_prob = tf.placeholder(tf.float32)

X = tf.reshape(_X, [-1, 10, 20])


def unit_lstm():
    lstm_cell = rnn.BasicLSTMCell(num_units=hidden_size, forget_bias=1.0, state_is_tuple=True)
    lstm_cell = rnn.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
    return lstm_cell

mlstm_cell = rnn.MultiRNNCell([unit_lstm() for i in range(3)], state_is_tuple=True)

init_state = mlstm_cell.zero_state(batch_size, dtype=tf.float32)
outputs, state = tf.nn.dynamic_rnn(mlstm_cell, inputs=X, initial_state=init_state, time_major=False)
h_state = outputs[:, -1, :]

W = tf.Variable(tf.truncated_normal([hidden_size, class_num], stddev=0.1), dtype=tf.float32)
bias = tf.Variable(tf.constant(0.1, shape=[class_num]), dtype=tf.float32)
y_pre = tf.nn.softmax(tf.matmul(h_state, W) + bias)

cross_entropy = -tf.reduce_mean(y * tf.log(y_pre))
train_op = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

sess.run(tf.global_variables_initializer())

# for i in range(1000):
#     _batch_size = 128
#     batch = mnist.train.next_batch(_batch_size)
#     print(batch)
#     print(batch[0])
#     print(batch[1])
#     print(len(batch[0]), len(batch[1]))
#     input()
#     if (i+1) % 200 == 0:
#         train_accuracy = sess.run(accuracy, feed_dict={
#             _X:batch[0], y:batch[1], keep_prob:1.0, batch_size:_batch_size
#         })
#         print('Iters %d, step %d, training accuracy %g' % (mnist.train.epochs_completed,
#                                                            (i+1), train_accuracy))
#     sess.run(train_op, feed_dict={_X:batch[0], y:batch[1], keep_prob:0.5, batch_size:_batch_size})
#
# print('test accuracy %g'% sess.run(accuracy, feed_dict={
#     _X:mnist.test.images, y:mnist.test.labels, keep_prob:1.0, batch_size:mnist.test.images.shape[0]
# }))



for i in range(200):
    _batch_size = 100
    k = random.randint(0, length /_batch_size)
    if (k+1)*_batch_size > length:
        continue
    train_X = train_f[k*_batch_size:(k+1)*_batch_size]
    train_Y = tra_y[k*_batch_size:(k+1)*_batch_size]

    if (i+1) % 50 == 0:
        train_accuracy = sess.run(accuracy, feed_dict={
            _X:train_X, y:train_Y, keep_prob:1.0, batch_size:_batch_size
        })
        print('step %d, training accuracy %g' %(i, train_accuracy))
    sess.run(train_op, feed_dict={_X:train_X, y:train_Y,
                                  keep_prob:1.0, batch_size:_batch_size})

print('test accuracy %g'% sess.run(accuracy, feed_dict={
     _X:train_f, y:tra_y, keep_prob:1.0, batch_size: len(train_y)
}))
import tensorflow as tf
from create_sentiment_featuresets import *
import numpy as np

train_x, train_y, test_x, test_y = create_feature_sets_and_labels('pos.txt', 'neg.txt')

num_sample = len(train_x[0])

x = tf.placeholder(tf.float32, [None, num_sample])
y = tf.placeholder(tf.float32)

n_notes_h1 = 500
n_notes_h2 = 500
n_notes_h3 = 500

batch_size = 100
n_classes = 2


def neural_network_model(data):

    hidden_layer_1 = {'weights': tf.Variable(tf.random_normal([num_sample, n_notes_h1])),
                      'biases': tf.Variable(tf.random_normal([n_notes_h1]))}

    hidden_layer_2 = {'weights': tf.Variable(tf.random_normal([n_notes_h1, n_notes_h2])),
                      'biases': tf.Variable(tf.random_normal([n_notes_h1]))}

    hidden_layer_3 = {'weights': tf.Variable(tf.random_normal([n_notes_h2, n_notes_h3])),
                      'biases': tf.Variable(tf.random_normal([n_notes_h1]))}

    output_layer = {'weights': tf.Variable(tf.random_normal([n_notes_h3, n_classes])),
                    'biases': tf.Variable(tf.random_normal([n_classes]))}

    l1 = tf.add(tf.matmul(data, hidden_layer_1['weights']),  hidden_layer_1['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_layer_2['weights']), hidden_layer_2['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_layer_3['weights']), hidden_layer_3['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

    return output


def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    hm_epochs = 20

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            i = 0
            while i < len(train_y):
                start = i
                end = i + batch_size
                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])
                i += batch_size

                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                epoch_loss += c
            print('epoch', epoch, 'completed out of ', hm_epochs, 'loss', epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        print('Accuracy', accuracy.eval({x: test_x, y: test_y}))


train_neural_network(x)

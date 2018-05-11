import tensorflow as tf

a = tf.constant(5)
b = tf.constant(6)

sess = tf.Session()
result = sess.run(tf.multiply(a, b))
print(result)


'==========================================================='


from tensorflow.examples.tutorials.mnist import input_data

"""
input > weight > hidden layer 1 (activation function ) > weight > hidden layer 2 (activation)> weight > output layer

compare output to intended input > cost or loss function (cross entropy)

optimization function (optimizer) > minimize cost (SGD ....)

backpropagation

feed forward + backprop = epoch 

"""

x = tf.placeholder(float, [None, 784])
y = tf.placeholder(float)

n_notes_h1 = 500
n_notes_h2 = 500
n_notes_h3 = 500

n_classes = 10


def neural_network_model(data):

    hidden_layer_1 = {'weights': tf.Variable(tf.random_normal(784, n_notes_h1)),
                      'biases': tf.Variable(tf.random_normal(n_notes_h1))}

    hidden_layer_2 = {'weights': tf.Variable(tf.random_normal(n_notes_h1, n_notes_h2)),
                      'biases': tf.Variable(tf.random_normal(n_notes_h1))}

    hidden_layer_3 = {'weights': tf.Variable(tf.random_normal(n_notes_h2, n_notes_h3)),
                      'biases': tf.Variable(tf.random_normal(n_notes_h1))}

    output_layer = {'weights': tf.Variable(tf.random_normal(n_notes_h3, n_classes)),
                    'biases': tf.Variable(tf.random_normal(n_classes))}

    l1 = tf.add(tf.matmul(data, hidden_layer_1['weights']) + hidden_layer_1['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_layer_2['weights']) + hidden_layer_2['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_layer_3['weights']) + hidden_layer_3['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

    return output








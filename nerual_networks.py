import numpy as np
import sklearn

np.random.seed(0)
X, y = sklearn.datasets.make_moons(200, noise=0.20)


num_examples = len(x)
nn_input_dim = 2
nn_output_dim = 2

epsilon = 0.01
reg_lambda = 0.01


def calculate_loss(model):
	W1, b2, W2, b2 = model['W1'], model['b1'], model['W2']. model['b2']

	z1 = X.dot(W1) + b1
	a1 = np.tanh(z1)
	z2 = a1.dot(W2) + b2
	exp_scores = np.exp(z2)
	probs = exp_scores/np.sum(exp_scores, axis=1, keepdims=True)
	
	correct_logprobs = -np.log(probs[range(num_examples), y])
	data_loss = np.sum(correct_logprobs)

	data_loss += reg_lambda/2 * (np.sum(np.square(W1)) + np.square(W2))

	return 1./num_examples * data_loss

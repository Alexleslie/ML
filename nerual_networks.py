import numpy as np
from sklearn import datasets


np.random.seed(0)

nn_input_dim = 2
nn_output_dim = 2

lr = 0.01
reg_lambda = 0.01


def calculate_loss(X, y, model):
    num_examples = len(X)
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']

    z1 = X.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores/np.sum(exp_scores, axis=1, keepdims=True)

    input(probs[range(num_examples), y])

    correct_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(correct_logprobs)

    data_loss += reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))

    return 1./num_examples * data_loss


def predict(model, x):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    z1 = x.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores/np.sum(exp_scores, axis=1, keepdims=True)
    return np.argmax(probs, axis=1)


def build_model(X, y, nn_hdim, num_passes=20000, print_loss=False):
    num_examples = len(y)
    np.random.seed(0)
    W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
    b1 = np.random.randn(1, nn_hdim)
    W2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
    b2 = np.random.randn(1, nn_output_dim)

    model = {}

    for i in range(0, num_passes):
        z1 = X.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2

        exp_scores = np.exp(z2)

        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        delta3 = probs
        delta3[range(num_examples), y] -= 1
        dW2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)

        dW2 += reg_lambda * W2
        dW1 += reg_lambda * W1

        W2 -= lr * dW2
        b2 -= lr * db2
        W1 -= lr * dW1
        b1 -= lr * db1

        model = {'W1': W1, 'W2': W2, 'b1': b1, 'b2': b2}

        if print_loss and i % 1000 == 0:
            print('Loss after iteration %d : %f' %(i, calculate_loss(X, y, model)))
    return model


if __name__ == '__main__':
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    data_x, data_y = datasets.make_moons(200,noise=0.1)
    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y)

    model = build_model(train_x, train_y, 10, print_loss=True)
    predictons = predict(model, test_x)

    print(accuracy_score(predictons, test_y))







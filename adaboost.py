from numpy import *
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def data_process(data_y):
    data_mat = []
    for i in data_y:
        if i == 0:
            data_mat.append(-1)
        else:
            data_mat.append(1)
    return (data_mat)


def stump_classify(data_matrix, dimen, thresh_val, thresh_Ineq):
    ret_array = ones((shape(data_matrix)[0], 1))
    if thresh_Ineq == 'lt':
        ret_array[data_matrix[:, dimen] <= thresh_val] = -1.0
    else:
        ret_array[data_matrix[:, dimen] > thresh_val] = -1.0
    return ret_array


def bulid_stump(data_arr, class_labels, D):
    data_mat = mat(data_arr); label_mat = mat(class_labels).T
    m, n = shape(data_mat)
    num_steps = 10.0; best_stump = {}; best_class = mat(zeros((m, 1)))
    min_error = inf
    for i in range(n):
        range_min = data_mat[:, i].min(); range_max = data_mat[:, i].max()
        step_size = (range_max - range_min) / num_steps
        for j in range(-1, int(num_steps)+1):
            for inequal in ['lt', 'gt']:
                thresh_val = (range_min + float(j) * step_size)
                predicted_vals = \
                    stump_classify(data_mat, i, thresh_val, inequal)
                err_arr = mat(ones((m, 1)))
                err_arr[predicted_vals == label_mat] = 0
                weighted_error = D.T*err_arr
                print('split: dim %d, thresh %.2f, thresh ineqal: '
                      '%s, the weighted error is %.3f' %
                      (i, thresh_val, inequal, weighted_error))
                if weighted_error < min_error:
                    min_error = weighted_error
                    best_class = predicted_vals.copy()
                    best_stump['dim'] = i
                    best_stump['thresh'] = thresh_val
                    best_stump['ineq'] = inequal
    return best_stump, min_error, best_class


def ada_boost(data_arr, class_labels, num_iters=40):
    weak_class_arr = []
    m = shape(data_arr)[0]
    D = mat(ones((m, 1))/m)
    agg_class_est = mat(zeros((m, 1)))
    for i in range(num_iters):
        best_stump, error, class_est = bulid_stump(data_arr, class_labels, D)
        print('D: ', D.T)
        alpha = float(0.5*log((1.0-error)/max(error, 1e-16)))
        best_stump['alpha'] = alpha
        weak_class_arr.append(best_stump)
        print('class_est: ', class_est.T)
        expon = multiply(-1*alpha*mat(class_labels).T, class_est)
        D = multiply(D, exp(expon))
        D = D/D.sum()
        agg_class_est += alpha * class_est
        print('agg_class_est: ', agg_class_est.T)

        agg_errors = multiply(sign(agg_class_est) !=
                              mat(class_labels).T, ones((m, 1)))
        error_rate = agg_errors.sum()/m
        print('total error : ', error_rate, '\n')
        if error_rate == 0.0: break
    return weak_class_arr


def ada_classify(dat_to_class, classifiter_arr):
    data_matrix = mat(dat_to_class)
    m = shape(data_matrix)[0]
    agg_class_est = mat(zeros((m, 1)))
    for i in range(len(classifiter_arr)):
        class_est = stump_classify(data_matrix, classifiter_arr[i]['dim'],
                                   classifiter_arr[i]['thresh'],
                                   classifiter_arr[i]['ineq'])
        agg_class_est += classifiter_arr[i]['alpha'] * class_est
    return sign(agg_class_est)


class AdaBoost:
    def fit(self, train_x, train_y):
        classifier_array = ada_boost(train_x, train_y)
        self.clf_arr = classifier_array

    def predict(self, test_x):
        prediction = []
        for i in test_x:
            predicts = ada_classify(i, self.clf_arr)
            prediction.append(predicts.min())
        return prediction


if __name__ == '__main__':
    data_x, data_y = load_iris().data[:100], load_iris().target[:100]
    data_y = data_process(data_y)
    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y)
    clf = AdaBoost()
    clf.fit(train_x, train_y)
    prediction = clf.predict(test_x)
    print(prediction)
    print(accuracy_score(test_y, prediction))



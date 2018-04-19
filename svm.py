import random
from sklearn.datasets import load_iris
import numpy as np


data_x, data_y = load_iris().data[:60], load_iris().target[:60]
y = np.zeros(len(data_y))
for i in range(len(data_y)):
    if i==1:
        y[i] =1
    else:
        y[i]=-1


def select_rand(i, m):
    j = i
    while j == i:
        j = int(random.uniform(0, m))
    return j


def clip_alpha(a, h, l):
    if a > h:
        a = h
    if l > a:
        a = l
    return a


def smo_simple(data_mat, class_label, C, toler, maxiter):
    data_mat = np.mat(data_mat); label_mat = np.mat(class_label).transpose()
    b = 0; m, n = np.shape(data_mat)
    alphas = np.mat(np.zeros((m, 1)))
    iter = 0
    while iter < maxiter:
        alpha_changed = 0
        for i in range(m):
            fxi = float(np.multiply(alphas, label_mat).T * (data_mat*data_mat[i, :].T)) + b
            Ei = fxi - float(label_mat[i])
            if((label_mat[i]*Ei < -toler)and(alphas[i] < C)) or \
            ((label_mat[i]*Ei > toler)and(alphas[i] > 0)):
                j = select_rand(i, m)
                fxj = float(np.multiply(alphas, label_mat).T * (data_mat*data_mat[j, :].T)) + b
                Ej = fxj - float(label_mat[j])
                alpha_i_old = alphas[i].copy()
                alpha_j_old = alphas[j].copy()
                if label_mat[i] != label_mat[j]:
                    L = max(0, alphas[i] - alphas[j])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H: print('L == H '); continue
                eta = 2.0 * data_mat[i, :]*data_mat[j, :].T - data_mat[i, :]*data_mat[i, :].T \
                    - data_mat[j, :] * data_mat[j, :].T
                if eta >= 0: print('eta >= 0'); continue
                alphas[j] -= label_mat[j]*(Ei - Ej)/eta
                alphas[j] = clip_alpha(alphas[j], H, L)
                if np.abs(alphas[j] - alpha_j_old) == 0.0001:
                    print(' j not moving enough')
                    continue
                alphas[i] += label_mat[j]*label_mat[i]*(alpha_j_old-alphas[j])
                b1 = b - Ei - label_mat[i]*(alphas[i]-alpha_i_old) * data_mat[i, :]*data_mat[i, :].T\
                    - label_mat[j]*(alphas[j] - alpha_j_old)*data_mat[i, :]*data_mat[j, :].T
                b2 = b - Ej - label_mat[i] * (alphas[i] - alpha_i_old) * data_mat[i, :] * data_mat[j, :].T \
                     - label_mat[j] * (alphas[j] - alpha_j_old)*data_mat[j, :]*data_mat[j, :].T
                if 0 < alphas[i] and C > alphas[i]: b = b1
                elif 0 < alphas[j] and C > alphas[j]: b = b2
                else: b = (b1+b2)/2
                alpha_changed +=1
                print('iter: %d i: %d, changed %d' % (iter, i, alpha_changed))
        if alpha_changed == 0:
            iter += 1
        else:
            iter = 0
        print('iteration number : %d' %  iter)
    return b, alphas

b, alpha = smo_simple(data_x, y, 1, 0.1, 400)
print(alpha[alpha>0])

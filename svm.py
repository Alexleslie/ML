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
        print('iteration number : %d' % iter)
    return b, alphas


if __name__ == '__main__':
    b, alpha = smo_simple(data_x, y, 1, 0.1, 400)
    print(alpha[alpha>0])



class Opt_Struct:
    def __init__(self, data_mat, class_labels, C, toler):
        self.x = data_mat
        self.label_mat = class_labels
        self.C = C
        self.tol = toler
        self.m = np.shape(data_mat)[0]
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        self.e_cache = np.mat(np.zeros(self.m, 2))


def calc_ek(os, k):
    fxk = float(np.multiply(os.alpha, os.label_mat).T *\
                (os.X*os.X[k, :].T)) + os.b
    Ek = fxk - float(os.label_mat[k])
    return Ek


def select_j(i, os, Ei):
    max_k = -1; max_deltae = 0; Ej = 0
    os.e_cache[i] = [1, Ei]
    valid_cache_list = np.nonzero(os.e_cache[:, 0].A)[0]
    if len(valid_cache_list) > 1:
        for k in valid_cache_list:
            if k == i: continue
            Ek = calc_ek(os, k)
            deltae = abs(Ei-Ek)
            if deltae > max_deltae:
                max_k = k; max_deltae = deltae; Ej = Ek
        return max_k, Ej
    else:
        j = select_rand(i, os.m)
        Ej = calc_ek(os, j)
    return j, Ej


def updateEk(os, k):
    Ek = calc_ek(os, k)
    os.e_cache[k] = [1, Ek]


def innerL(i ,os):
    Ei = calc_ek(os, i)
    if (os.label_mat[i]*E[i] < -os.tol and os.alphas[i] < os.C) or \
            (os.label_mat[i]*E[i] > os.tol and os.alphas[i] > 0):
        j, Ej = select_j(i, os, Ei)
        alpha_i_old = os.alphas[i].copy()
        alpha_j_old = os.alphas[j].copy()
        if os.label_mat[i] != os.label_mat[j]:
            L = max(0, os.alphas[i] - os.alphas[j])
            H = min(os.C, os.C + os.alphas[j] - os.alphas[i])
        else:
            L = max(0, os.alphas[j] + os.alphas[i] - os.C)
            H = min(os.C, os.alphas[j] + os.alphas[i])
        if L==H: print('L==H'); return 0
        eta = 2.0 * os.X[i, :] * os.X[j, :].T - os.X[i, :] * os.X[i, :].T \
              - os.X[j, :] * os.X[j, :].T
        if eta >=0: print('eta>=0'); return 0
        os.alphas[j] -= os.label_mat[j] * (Ei - Ej) / eta
        os.alphas[j] = clip_alpha(os.alphas[j], H, L)
        updateEk(os, j)
        if abs(os.alphas[j] -alpha_j_old ) < 0.00001:
            print('j not moving enough'); return 0
        os.alphas[i] += os.label_mat[j] * os.label_mat[i] * (alpha_j_old - os.alphas[j])
        updateEk(os, i)
        b1 = os.b - Ei - os.label_mat[i] * (os.alphas[i] - alpha_i_old) * os.X[i, :] * os.X[i, :].T \
             - os.label_mat[j] * (os.alphas[j] - alpha_j_old) * os.X[i, :] * os.X[j, :].T
        b2 = os.b - Ej - os.label_mat[i] * (os.alphas[i] - alpha_i_old) * os.X[i, :] * os.X[j, :].T \
             - os.label_mat[j] * (os.alphas[j] - alpha_j_old) * os.X[j, :] * os.X[j, :].T
        if 0 < os.alphas[i] and os.C > os.alphas[i]:
            os.b = b1
        elif 0 < os.alphas[j] and C > os.alphas[j]:
            os.b = b2
        else:
            os.b = (b1 + b2) / 2
        return 1
    else:
        return 0
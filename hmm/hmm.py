import numpy as np
from sklearn.externals import joblib

with open('xss-train.txt', encoding='utf-8') as f:
    content = f.readlines()

with open('good-xss.txt', encoding='utf-8') as f:
    good_query = f.readlines()
    good_query = good_query[:2000]

content = content + good_query


def et1(str):
    vers = []
    for c in str:
        c = c.lower()
        if (ord(c) >= ord('a')) and (ord(c) <= ord('z')):
            vers.append([ord('A')])
        elif (ord(c) >= ord('0')) and (ord(c) <= ord('9')):
            vers.append([ord('N')])
        else:
            vers.append([ord('C')])
    return np.array(vers)

X = [[0]]
X_lens = [1]

y_good = [0 for _ in range(2000)]
y_bad = [1 for _ in range(2000)]

for i in content:
    X = np.concatenate([X, et1(i)])
    X_lens.append(len(i))

from hmmlearn import hmm

# remodel = hmm.GaussianHMM(n_components=2, covariance_type='full', n_iter=100)
# remodel.fit(X, X_lens)

# joblib.dump(remodel, 'hmm.m')
# print('ok')

remodel = joblib.load('hmm.m')
start = 0

score_list = []

X_all = []
for i in X_lens:
    end = start + i
    X_all.append(X[start:end])
    start = end

X_all = X_all[1:]
y_all = y_bad + y_good

prediction = []
print(len(X_all))
input()
for i in X_all:
    score = remodel.score(i)
    if score < 300:
        prediction.append(0)
    else:
        prediction.append(1)

from sklearn.metrics import accuracy_score

print(accuracy_score(y_all, prediction))
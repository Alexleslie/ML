import numpy as np
from sklearn.externals import joblib
import random
import pandas
from urllib.parse import urlparse, unquote, parse_qsl

with open('xss-train.txt', encoding='utf-8') as f:
    content = [i.strip('\n') for i in f.readlines()]


good_query = pandas.read_csv('normal_examples.csv')


good_query = ['php?'+ unquote(i) for i in good_query['url'] if (len(i) < 200) and (len(i)>1)]
#
# with open('good-xss.txt', encoding='utf-8') as f:
#     good_query = [i.strip('\n') for i in f.readlines()]
#     random.shuffle(good_query)
#content = content + good_query
random.shuffle(good_query)

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


def deal(url):
    from urllib.parse import urlparse, unquote, parse_qsl
    result = urlparse(url)
    query = unquote(result.query)
    params = parse_qsl(query, True)
    vers = []
    for k, v in params:
        vers.append(v)
    return vers

good = []
bad = []

for i in good_query:
    if len(good) < 2000:
        if deal(i):
            good.append(deal(i))
        else:
            continue

for i in content:
    if len(bad) < 2000:
        bad.append(deal(i))


def conbine(url_list):
    conbine_list = []
    for i in url_list:
        if len(i) <= 1:
            conbine_list += i
        else:
            string = ''
            for j in i:
                string += j
            conbine_list.append(string)
    return conbine_list

Good_conbine = conbine(good)
Bad_conbine = conbine(bad)


X = [[0]]
X_lens = [1]

y_good = [0 for _ in range(2000)]
y_bad = [1 for _ in range(2000)]

content = Good_conbine + Bad_conbine

for i in content:
    print(i)
    X = np.concatenate([X, et1(i)])
    X_lens.append(len(i))

from hmmlearn import hmm

remodel = hmm.GaussianHMM(n_components=2, covariance_type='full', n_iter=100)
remodel.fit(X, X_lens)

joblib.dump(remodel, 'hmm.m')
print('ok')
input()

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

for i in X_all:
    score = remodel.score(i)
    if score < 300:
        prediction.append(0)
    else:
        prediction.append(1)

from sklearn.metrics import accuracy_score

print(accuracy_score(y_all, prediction))
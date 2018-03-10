import numpy as np
import xgboost as xgb
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cross_validation import KFold


SEED = 0
NFOLDS = 5

rf_params = {
    'n_jobs': -1,
    'n_estimators': 500,
    'warm_start': True,
    'max_depth': 6,
    'min_samples_leaf': 2,
    'max_features': 'sqrt',
    'verbose': 0
}

et_params = {
    'n_jobs': -1,
    'n_estimators': 500,
    'max_depth': 8,
    'min_samples_leaf': 2,
    'verbose': 0
}

ada_params = {
    'n_estimators': 500,
    'learning_rate': 0.75
}

gb_params = {
    'n_estimators': 500,
    'max_depth': 5,
    'min_samples_leaf': 2,
    'verbose': 0
}

svc_params = {
    'kernel': 'linear',
    'C': 1
    }


class sklearn_helper():
    def __init__(self, clf, seed=0, params=None):
        try:
            params['random_state'] = seed
            self.clf = clf(**params)
        except:
            self.clf = clf()

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x_test):
        return self.clf.predict(x_test)

    def fit(self, x, y):
        return self.fit(x, y)

    def feature_importances(self, x, y):
        print(self.clf.fit(x, y).feature_importances_)


def get_oof(clf, x_train, y_train, x_test):
    print('finish one classifier, having five classifier')
    n_train = x_train.shape[0]
    n_test = x_test.shape[0]

    kf = KFold(n_train, n_folds=NFOLDS, random_state=SEED)

    oof_train = np.zeros((n_train, ))
    oof_test = np.zeros((n_test, ))
    oof_test_skf = np.empty((NFOLDS, n_test))

    for i, (train_index, test_index) in enumerate(kf):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)

    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


def stack_classifier(x_train, y_train, x_test, no_clf_list=''):

    rf = sklearn_helper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
    et = sklearn_helper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
    ada = sklearn_helper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)
    gb = sklearn_helper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)
    svc = sklearn_helper(clf=SVC, seed=SEED, params=svc_params)
    nw = sklearn_helper(clf=MLPClassifier, params={'max_iter': 400})
    bg = sklearn_helper(clf=BaggingClassifier, params={})

    clf_list = {'rf': rf, 'et': et, 'ada': ada, 'gb': gb, 'svc': svc, 'nw': nw, 'bg': bg}
    for i in no_clf_list:
        del(clf_list[i])

    train_list = []
    test_list = []

    for i in clf_list.values():
        oof_train, oof_test = get_oof(i, x_train, y_train, x_test)
        train_list.append(oof_train)
        test_list.append(oof_test)

    x_train = np.concatenate((train_list), axis=1)
    x_test = np.concatenate((test_list), axis=1)

    gbm = xgb.XGBClassifier(
        n_estimators= 2000,
        max_depth= 4,
        min_child_weight= 2,
        gamma=0.9,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        nthread=-1,
        scale_pos_weight=1).fit(x_train, y_train)

    predictions = gbm.predict(x_test)

    return predictions

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.ensemble import RandomForestRegressor

row_x_train = pd.DataFrame(pd.read_csv('train.csv'))
row_x_test = pd.DataFrame(pd.read_csv('test.csv'))

PassengerId = row_x_test['PassengerId']

full_data = [row_x_train, row_x_test]


def set_cabin_and_name(df):
    df['Name_length'] = df['Name'].apply(len)
    df['Cabin_has'] = df['Cabin'].apply(lambda x: 0 if type(x) == float else 1)
    return df


def multi_fea(dataset):
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
    dataset['Fare'] = dataset['Fare'].fillna(row_x_train['Fare'].median())
    return dataset


def set_missing_ages(df):
    df.loc[(df.Fare.isnull()), 'Fare'] = 0
    age_df = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]

    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()

    y = known_age[:, 0]
    X = known_age[:, 1:]

    rfr = RandomForestRegressor(n_estimators=100)
    rfr.fit(X, y)

    predicted_ages = rfr.predict(unknown_age[:, 1:])
    df.loc[(df.Age.isnull()), 'Age'] = predicted_ages

    return df


def get_title(name):
    title_search = re.search('([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ''


def name_deal(data):
    data['Title'] = data['Name'].apply(get_title)
    data['Title'] = data['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major',
                                            'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    data['Title'] = data['Title'].replace('Mlle', 'Miss')
    data['Title'] = data['Title'].replace('Ms', 'Miss')
    data['Title'] = data['Title'].replace('Mme', 'Mrs')
    return data


def first_deal_all(df):
    df['Sex'] = df['Sex'].map({'female': 0, 'male': 1}).astype(int)

    title_mapping = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Rare': 5}
    df['Title'] = df['Title'].map(title_mapping)
    df['Title'] = df['Title'].fillna(0)
    df['Title'] = df['Title'].astype(int)

    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

    df.loc[df['Fare'] <= 7.91, 'Fare'] = 0
    df.loc[(df['Fare'] > 7.91) & (df['Fare'] < 14.454), 'Fare'] = 1
    df.loc[(df['Fare'] > 14.454) & (df['Fare'] < 31), 'Fare'] = 2
    df.loc[df['Fare'] > 31, 'Fare'] = 3
    df['Fare'] = df['Fare'].astype(int)

    df.loc[df['Age'] <= 16, 'Age'] = 0
    df.loc[(df['Age'] > 16) & (df['Age'] <= 32), 'Age'] = 1
    df.loc[(df['Age'] > 32) & (df['Age'] <= 48), 'Age'] = 2
    df.loc[(df['Age'] > 48) & (df['Age'] <= 64), 'Age'] = 3
    df.loc[df['Age'] > 64, 'Age'] = 4
    df['Age'] = df['Age'].astype(int)

    return df


def all_deal_data(row_data):
    row = set_cabin_and_name(row_data)
    row = set_missing_ages(row)
    row = multi_fea(row)
    row = name_deal(row)
    row = first_deal_all(row)

    drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp']
    data = row.drop(drop_elements, axis=1)

    return data


def plot_learning_curve(clf, title, X, y, ylim=None, cv=None, n_jobs=3, train_sizes=np.linspace(.05, 1., 5)):
    train_sizes, train_scores, test_scores = learning_curve(
        clf, X, y, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    ax = plt.figure().add_subplot(111)
    ax.set_title(title)
    if ylim is not None:
        ax.ylim(*ylim)
    ax.set_xlabel(u"train_num_of_samples")
    ax.set_ylabel(u"score")

    ax.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
                    alpha=0.1, color="b")
    ax.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std,
                    alpha=0.1, color="r")
    ax.plot(train_sizes, train_scores_mean, 'o-', color="b", label=u"train score")
    ax.plot(train_sizes, test_scores_mean, 'o-', color="r", label=u"testCV score")

    ax.legend(loc="best")
    plt.show()
    midpoint = ((train_scores_mean[-1] + train_scores_std[-1]) + (test_scores_mean[-1] - test_scores_std[-1])) / 2
    diff = (train_scores_mean[-1] + train_scores_std[-1]) - (test_scores_mean[-1] - test_scores_std[-1])
    return midpoint, diff


X_train = all_deal_data(row_x_train)
X_test = all_deal_data(row_x_test)

y_train = X_train['Survived'].ravel()
x_train = X_train.drop(['Survived'], axis=1).values
x_test = X_test.values

from stack_first import stack_classifier

predictions = stack_classifier(x_train, y_train, x_test)

print("Training is complete")


StackingSubmission = pd.DataFrame({ 'PassengerId': PassengerId,
                            'Survived': predictions})
StackingSubmission.to_csv("StackingSubmission.csv", index=False)

#plot_learning_curve(gbm, u"learning_rate", X=x_train, y=y_train)
#print(metrics.accuracy_score(y_test, y_pred))


#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 10:09:24 2015

@author: ddboline
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import gzip
import cPickle as pickle

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.cross_validation import train_test_split

from sklearn.metrics import mean_squared_error

from load_data import load_data

def transform_to_log(y):
    return np.log1p(y)

def transform_from_log(ly):
    return np.expm1(ly)

def train_model_parallel(xtrain, ytrain, index=0):
    xTrain, xTest, yTrain, yTest = train_test_split(xtrain, ytrain[:, index],
                                                    test_size=0.25)
#    model = RandomForestRegressor()
#    model = LogisticRegression()
    model = GradientBoostingRegressor(verbose=1)

    n_est = [10, 100, 200]
    m_dep = [5, 10, 40]

    model = GridSearchCV(estimator=model,
                                param_grid=dict(n_estimators=n_est, max_depth=m_dep),
                                scoring=scorer,
                                n_jobs=-1, verbose=1)

    model.fit(xTrain, yTrain)
    ypred = model.predict(xTest)
    print('score %d %s' % (index, model.score(xTest, yTest)))
    print('RMSLE %d %s' % (index, np.sqrt(mean_squared_error(yTest, ypred))))
    with gzip.open('model_%d.pkl.gz', 'wb') as pklfile:
        pickle.dump(model, pklfile, protocol=2)
    return

def test_model_parallel(xtrain, ytrain):
    xTrain, xTest, yTrain, yTest = train_test_split(xtrain, ytrain,
                                                    test_size=0.25)
    ypred = np.zeros((yTest.shape[0], 3))
    for idx in range(3):
        with gzip.open('model_%d.pkl.gz', 'rb') as pklfile:
            model = pickle.load(pklfile)
        ypred[:, idx] = model.predict(xTest)
    print('RMSLE %s' % np.sqrt(mean_squared_error(yTest, ypred)))
    return

def prepare_submission_parallel(xtest, ytest):
    YLABELS = ['*', '**', '***']
    for idx in range(3):
        with gzip.open('model_%d.pkl.gz', 'rb') as pklfile:
            model = pickle.load(pklfile)
        ypred = transform_from_log(model.predict(xtest)).astype(int)
        print(ypred.shape, ytest.shape)
#        ytest[:, YLABELS[idx]] = 
    print(ytest.shape)
    ytest.to_csv('submission.csv', index=False)
    return

def my_model(index=0):
    xtrain, ytrain, xtest, ytest = load_data()
    
    ytrain = transform_to_log(ytrain)
    
#    for idx in range(3):
#        train_model_parallel(xtrain, ytrain, index=idx)

#    test_model_parallel(xtrain, ytrain)
    prepare_submission_parallel(xtest, ytest)

    return

if __name__ == '__main__':
    my_model()

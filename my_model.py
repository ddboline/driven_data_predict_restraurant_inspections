#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 10:09:24 2015

@author: ddboline
"""
#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function
#from __future__ import unicode_literals

import os
import gzip
import cPickle as pickle

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import mean_squared_error

from load_data import load_data

NCPU = len([x for x in open('/proc/cpuinfo').read().split('\n')
            if x.find('processor') == 0])

def transform_to_log(y):
    return np.log1p(y)

def transform_from_log(ly):
    return np.round(np.expm1(ly)).astype(int)

def scorer(estimator, X, y):
    ypred = estimator.predict(X)
    return 1.0/mean_squared_error(ypred, y)

def train_model_parallel_xgb(xtrain, ytrain, index=0, depth=2):
    import xgboost as xgb
    xTrain, xTest, yTrain, yTest = train_test_split(xtrain, ytrain[:, index],
                                                    test_size=0.25)
    dtrain = xgb.DMatrix(xTrain, label=yTrain)
    dtest = xgb.DMatrix(xTest, label=yTest)

    param = {'bst:max_depth':depth,
             'bst:eta':1, 
             'silent':1, 
             'objective':'reg:linear' }
    param['nthread'] = NCPU
    plst = param.items()
    plst += [('eval_metric', 'rmse')] # Multiple evals can be handled in this way
    
    evallist  = [(dtest,'eval'), (dtrain,'train')]
    num_round = 100
    model = xgb.train(plst, dtrain, num_round, evallist,
                    early_stopping_rounds=10)
    with gzip.open('model_bst_%d.pkl.gz' % index, 'wb') as pklfile:
        pickle.dump(model, pklfile, protocol=2)

def test_model_parallel_xgb(xtrain, ytrain):
    import xgboost as xgb
    print('start test_model_parallel_xgb')
    xTrain, xTest, yTrain, yTest = train_test_split(xtrain, ytrain,
                                                    test_size=0.25)
    ypred = np.zeros((yTest.shape[0], 3))
    dtest = xgb.DMatrix(xTest)
    for idx in range(3):
        with gzip.open('model_bst_%d.pkl.gz' % idx, 'rb') as pklfile:
            model = pickle.load(pklfile)
        ypred[:, idx] = model.predict(dtest, ntree_limit=model.best_iteration)
    print('\nRMSLE %s\n' % np.sqrt(mean_squared_error(yTest, ypred)))
    return

def prepare_submission_parallel_xgb(xtest, ytest):
    import xgboost as xgb
    print('start prepare_submission_parallel_xgb')
    YLABELS = [u'*', u'**', u'***']
    print(ytest.columns)
    dtest = xgb.DMatrix(xtest)
    for idx in range(3):
        model = xgb.Booster({'nthread':NCPU})
        with gzip.open('model_bst_%d.pkl.gz' % idx, 'rb') as pklfile:
            model = pickle.load(pklfile)
        key = YLABELS[idx]
        ytest.loc[:, key] = transform_from_log(model.predict(dtest,
                                               ntree_limit=model.best_iteration))
    print(ytest.shape)
    with gzip.open('submission.csv.gz', 'wb') as subfile:
        ytest.to_csv(subfile, index=False)
    return


def train_model_parallel(xtrain, ytrain, index=0):
    xTrain, xTest, yTrain, yTest = train_test_split(xtrain, ytrain[:, index],
                                                    test_size=0.25)
#    model = RandomForestRegressor()
#    model = LogisticRegression()
    model = GradientBoostingRegressor(verbose=1)

    n_est = [10, 50]
    m_dep = [5, 3]

    model = GridSearchCV(estimator=model,
                                param_grid=dict(n_estimators=n_est,
                                                max_depth=m_dep),
                                scoring=scorer,
                                n_jobs=-1, verbose=1)

    model.fit(xTrain, yTrain)
    ypred = model.predict(xTest)
    if hasattr(model, 'best_params_'):
        print('best_params', model.best_params_)
    print('score %d %s' % (index, model.score(xTest, yTest)))
    print('RMSLE %d %s' % (index, np.sqrt(mean_squared_error(yTest, ypred))))
    with gzip.open('model_%d.pkl.gz' % index, 'wb') as pklfile:
        pickle.dump(model, pklfile, protocol=2)
    return

def test_model_parallel(xtrain, ytrain):
    xTrain, xTest, yTrain, yTest = train_test_split(xtrain, ytrain,
                                                    test_size=0.25)
    ypred = np.zeros((yTest.shape[0], 3))
    for idx in range(3):
        with gzip.open('model_%d.pkl.gz' % idx, 'rb') as pklfile:
            model = pickle.load(pklfile)
        ypred[:, idx] = model.predict(xTest)
    rmsle = np.sqrt(mean_squared_error(yTest, ypred))
    print('RMSLE %s' % rmsle)
    return rmsle

def prepare_submission_parallel(xtest, ytest):
    YLABELS = [u'*', u'**', u'***']
    print(ytest.columns)
    for idx in range(3):
        with gzip.open('model_%d.pkl.gz' % idx, 'rb') as pklfile:
            model = pickle.load(pklfile)
        key = YLABELS[idx]
        ytest.loc[:, key] = transform_from_log(model.predict(xtest))
    print(ytest.shape)
    with gzip.open('submission.csv.gz', 'wb') as subfile:
        ytest.to_csv(subfile, index=False)
    return

def my_model(index=0):
    xtrain, ytrain, xtest, ytest = load_data()

    ytrain = transform_to_log(ytrain)

    results = {}
    for depth in range(2,10):
        for idx in range(3):
            train_model_parallel_xgb(xtrain, ytrain, index=idx, depth=depth)

        rmsle = test_model_parallel_xgb(xtrain, ytrain)
        prepare_submission_parallel_xgb(xtest, ytest)
        os.rename('submission.csv.gz', 'submission_dep_%d_20150603.csv.gz' % depth)
        results[depth] = rmsle
    
    for depth in sorted(results):
        print(depth, results[depth])

    return

if __name__ == '__main__':
    my_model()

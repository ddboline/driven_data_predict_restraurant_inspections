#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 13:48:58 2015

@author: ddboline
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import pandas as pd

from dateutil.parser import parse

NEIGHBORHOODS = [u'Allston/Brighton', u'Back Bay', u'Beacon Hill',
                 u'Charlestown', u'Chinatown', u'Dorchester', u'Downtown',
                 u'Dudley Square', u'East Boston', u'Egleston Square',
                 u'Fenway', u'Fields Corner', u'Financial District',
                 u'Hyde Park', u'Jamaica Plain', u'Leather District',
                 u'Mattapan', u'Mission Hill', u'North End', u'Roslindale',
                 u'Roslindale Village', u'South Boston', u'South End',
                 u'Uphams Corner', u'Waterfront', u'West Roxbury',
                 u'West Roxbury Center']

CITIES = ['Boston', 'Hyde Park', 'West Roxbury', 'Dorchester', 'Roslindale', 
          'Roxbury', 'Charlestown', 'Brighton', 'Jamaica Plain', 
          'Dorchester Center', 'Allston', 'East Boston', 'South Boston', 
          'Roxbury Crossing', 'Mattapan', 'Mission Hill', 'Chestnut Hill']

def clean_data(df, do_plots=False):
    df['city'] = df['city'].map({k: i for (i, k) in enumerate(CITIES)})
    df.loc[df['city'].isnull(), 'city'] = -1
    df['neighborhoods'] = df['neighborhoods'].map({k: i for (i, k) in 
                                                   enumerate(NEIGHBORHOODS)})
    df.loc[df['neighborhoods'].isnull(), 'neighborhoods'] = -1
    df['neighborhoods'] = df['neighborhoods'].astype(int)
    for col in ['delivery', 'takeout', 'good for_dessert', 
                'good for_latenight', 'good for_lunch', 'good for_dinner', 
                'good for_brunch', 'good for_breakfast', 'groups', 'byob', 
                'parking_garage', 'parking_street', 'parking_validated', 
                'parking_lot', 'parking_valet', 'tv', 'outdoor', 'reserve', 
                'ambience_romantic', 'ambience_intimate', 'ambience_touristy', 
                'ambience_hipster', 'ambience_divey', 'ambience_classy', 
                'ambience_trendy', 'ambience_casual', 'waiter', 'kids', 
                'wheelchair']:
        df[col] = df[col].map({'0': 0, 'False': 0, 'True': 1})
    for col in df.columns:
        if 'checkin' in col:
            df.loc[df[col].isnull(), col] = 0
            df[col] = df[col].astype(int)
    df['creditcard'] = df['creditcard'].map({'0': 0, 'False': 0, 'True': 1, 
                                             '{}': 0})
    df.loc[df['open'].isnull(), 'open'] = -1
    df['open'] = df['open'].astype(int)

    for col in ['votes_funny', 'votes_useful', 'votes_cool', 'w_stars', 
                'stars']:
        df[col] = df[col].astype(np.float64)
        df[col] = np.divide(df[col], df['n_review'])

    df['date'] = df['date'].apply(lambda x: parse(x))
    df['year'] = df['date'].apply(lambda x: x.year)
    df['month'] = df['date'].apply(lambda x: x.month)
    df['weekday'] = df['date'].apply(lambda x: x.weekday())

    for col in df.columns:
        if df[col].isnull().sum() > 0 and df[col].dtype != object:
            df.loc[df[col].isnull(), col] = np.mean(df[df[col].notnull()][col])
    
    if do_plots:
        df = df.rename(columns={'*': 'minor', '**': 'major', '***': 'severe',
                            'byob/corkage_yes_corkage': 'corkage_yes_corkage',
                            'byob/corkage_yes_free': 'corkage_yes_free',
                            'byob/corkage_no': 'corkage_no'})
    
    df = df.drop(labels=['ambience_upscale', 'latitude', 'longitude'], 
                 axis=1)
    return df

def load_data(do_plots=False):
    train_df = pd.read_csv('train.csv.gz', compression='gzip', 
                           low_memory=False)
    test_df = pd.read_csv('test.csv.gz', compression='gzip', low_memory=False)

    train_df = clean_data(train_df, do_plots)
    test_df = clean_data(test_df, do_plots)
    
    for col in train_df.columns:
        if train_df[col].dtype != test_df[col].dtype:
            print(col, train_df[col].dtype, test_df[col].dtype, 
                  test_df[test_df[col].isnull()].shape)
    
    if do_plots:
        from plot_data import plot_data
        plot_data(train_df.drop(labels=['date', 'id', 'restaurant_id', 'name'], 
                                axis=1), prefix='train')
        plot_data(test_df.drop(labels=['date', 'id', 'restaurant_id', 'name'], 
                               axis=1), prefix='test')
    
    print(train_df['w_stars'].describe(), train_df['stars'].describe())
    print(train_df.shape, test_df.shape)
    
    xtrain = train_df.drop(labels=['*', '**', '***', 'restaurant_id', 
                                   'name', 'date'], axis=1).values
    ytrain = train_df[['*', '**', '***']].values
    xtest = test_df.drop(labels=['*', '**', '***', 'restaurant_id', 
                                   'name', 'date'], axis=1).values
    ytest = test_df[['id', 'date', 'restaurant_id', '*', '**', '***']]
    
    print(xtrain.shape, ytrain.shape, xtest.shape, ytest.shape)
    return xtrain, ytrain, xtest, ytest

if __name__ == '__main__':
    xtrain, ytrain, xtest, ytest = load_data(do_plots=True)

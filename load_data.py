#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 13:48:58 2015

@author: ddboline
"""
import pandas as pd

NEIGHBORHOODS = [u'Allston/Brighton', u'Back Bay', u'Beacon Hill',
                 u'Charlestown', u'Chinatown', u'Dorchester', u'Downtown',
                 u'Dudley Square', u'East Boston', u'Egleston Square',
                 u'Fenway', u'Fields Corner', u'Financial District',
                 u'Hyde Park', u'Jamaica Plain', u'Leather District',
                 u'Mattapan', u'Mission Hill', u'North End', u'Roslindale',
                 u'Roslindale Village', u'South Boston', u'South End',
                 u'Uphams Corner', u'Waterfront', u'West Roxbury',
                 u'West Roxbury Center']

def load_data():
    train_df = pd.read_csv('train_labels.csv.gz', compression='gzip')
    id_yelp_map_df = pd.read_csv('restaurant_ids_to_yelp_ids.csv.gz',
                                 compression='gzip')
    submit_df = pd.read_csv('SubmissionFormat.csv.gz', compression='gzip')
#    submit_df = pd.read_csv('PhaseIISubmissionFormat.csv.gz',
#                            compression='gzip')

    print train_df.columns
    print id_yelp_map_df.columns
    print submit_df.columns

    for col in id_yelp_map_df.columns:
        print col, len(id_yelp_map_df[col].unique())
    return

if __name__ == '__main__':
    load_data()

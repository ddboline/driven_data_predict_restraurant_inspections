#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 09:58:47 2015

@author: ddboline
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from nltk.corpus import stopwords

import re
import gzip
import json

from sklearn.feature_extraction.text import CountVectorizer

import numpy as np
import pandas as pd

STOPS = set(stopwords.words('english'))

def text_to_wordlist(txt):
    txt = re.sub("[^a-zA-Z0-9]", " ", txt)
    words = []
    for word in txt.lower().split():
        if word not in STOPS:
            words.append(word)
    return ' '.join(words)

def text_features():
    train_df = pd.read_csv('train.csv.gz', compression='gzip',
                           low_memory=False)
    test_df = pd.read_csv('test.csv.gz', compression='gzip', low_memory=False)
    id_yelp_map_df = pd.read_csv('restaurant_ids_to_yelp_ids.csv.gz',
                                 compression='gzip')

    yelp_id_map = {}
    for idx, row in id_yelp_map_df.iterrows():
        yelp_id_map[row['yelp_id_0']] = row['restaurant_id']
        for ydx in range(4):
            col_ = 'yelp_id_%d' % ydx
            if row[col_]:
                yelp_id_map[row[col_]] = row['restaurant_id']

    id_to_ncomplaint = {}
    for idx, row in train_df.iterrows():
        rid = row['restaurant_id']
        ncomplaint = row['*'] + row['**'] + row['***']
        id_to_ncomplaint[rid] = ncomplaint

    word_list = []
    bid_list = []
    fname = 'yelp_academic_dataset_review.json.gz'
    with gzip.open('yelp_boston_academic_dataset/%s' % fname, 'rb') as infile:
        for line in infile:
            out = json.loads(line)
            if out['business_id'] not in yelp_id_map:
                continue
            rid = yelp_id_map[out['business_id']]
            if rid not in id_to_ncomplaint:
                continue
            word_list.append(text_to_wordlist(out['text']))
            bid_list.append(rid)

    nfeatures = 1000
    vectorizer = CountVectorizer(analyzer='word', max_features=nfeatures)
    review_vector = vectorizer.fit_transform(word_list).toarray()
    review_dict = {bid_list[idx]: idx for idx in range(len(bid_list))}
    train_review_vector = np.zeros((train_df.shape[0], nfeatures), dtype=int)
    test_review_vector = np.zeros((test_df.shape[0], nfeatures), dtype=int)
    for idx in range(train_df.shape[0]):
        rid = train_df.loc[idx, 'restaurant_id']
        if rid in review_dict:
            train_review_vector[idx, :] = review_vector[review_dict[rid], :]
    for idx in range(test_df.shape[0]):
        rid = test_df.loc[idx, 'restaurant_id']
        if rid in review_dict:
            test_review_vector[idx, :] = review_vector[review_dict[rid], :]

    for idx in range(nfeatures):
        train_df['review_vec_%02d' % idx] = train_review_vector[:, idx]
        test_df['review_vec_%02d' % idx] = test_review_vector[:, idx]

    with gzip.open('train_final.csv.gz', 'wb') as outfile:
        train_df.to_csv(outfile, index=False)
    with gzip.open('test_final.csv.gz', 'wb') as outfile:
        test_df.to_csv(outfile, index=False)

    return

if __name__ == '__main__':
    text_features()

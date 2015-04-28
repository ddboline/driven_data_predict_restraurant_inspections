#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 14:12:38 2015

@author: ddboline
"""
import pandas as pd
import gzip
import json
from collections import defaultdict

def feature_extraction():
    train_df = pd.read_csv('train_labels.csv.gz', compression='gzip')
    id_yelp_map_df = pd.read_csv('restaurant_ids_to_yelp_ids.csv.gz',
                                 compression='gzip')

    yelp_id_map = {}
    for idx, row in id_yelp_map_df.iterrows():
        yelp_id_map[row['yelp_id_0']] = row['restaurant_id']
        for ydx in range(4):
            col = 'yelp_id_%d' % ydx
            if row[col]:
                yelp_id_map[row[col]] = row['restaurant_id']

    id_to_ncomplaint = {}
    for idx, row in train_df.iterrows():
        rid = row['restaurant_id']
        ncomplaint = row['*'] + row['**'] + row['***']
        id_to_ncomplaint[rid] = ncomplaint

    print train_df.columns
    print id_yelp_map_df.columns
    
    for fname in ('yelp_academic_dataset_business.json.gz', 
                  'yelp_academic_dataset_review.json.gz',
                  'yelp_academic_dataset_user.json.gz',
                  'yelp_academic_dataset_checkin.json.gz',
                  'yelp_academic_dataset_tip.json.gz',):
        print '\n\n%s' % fname
        with gzip.open('yelp_boston_academic_dataset/%s' % fname, 'rb')\
                as infile:
            for line in infile:
                out = json.loads(line)
                for k, v in sorted(out.items()):
                    print '%s: %s' % (k, v)
                break

    fname = 'yelp_academic_dataset_business.json.gz'
    with gzip.open('yelp_boston_academic_dataset/%s' % fname, 'rb') as infile:
        attr_keys = defaultdict(int)
        cat_types = defaultdict(int)
        neighborhoods = defaultdict(int)
        for line in infile:
            out = json.loads(line)
            if out['business_id'] not in yelp_id_map:
                continue
            rid = yelp_id_map[out['business_id']]
            if rid not in id_to_ncomplaint:
                continue
            count = id_to_ncomplaint[rid]
            if count == 0:
                continue
            for key in out['attributes'].keys():
                attr_keys[key] += 1
            for cat in out['categories']:
                cat_types[cat] += 1
            for neighb in out['neighborhoods']:
                neighborhoods[neighb] += 1
        print '\n\nAttributes:', len(attr_keys)
        for k, v in sorted(attr_keys.items(), key=lambda x: x[1])[-30:]:
            print '%s: %s' % (k, v)
        print '\n\nCategories:', len(cat_types)
        for k, v in sorted(cat_types.items(), key=lambda x: x[1])[-30:]:
            print '%s: %s' % (k, v)
        print '\n\nNeighborhoods:', len(neighborhoods)
        for k, v in sorted(neighborhoods.items(), key=lambda x: x[1])[-30:]:
            print '%s: %s' % (k, v)
    return

if __name__ == '__main__':
    feature_extraction()

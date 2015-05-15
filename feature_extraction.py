#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 14:12:38 2015

@author: ddboline
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import re
import csv
import gzip
import json
import datetime
from dateutil.parser import parse

import pandas as pd

from collections import defaultdict

YEAR = datetime.date.today().year
MONTH = datetime.date.today().month


CATEGORIES = [u'American (New)', u'American (Traditional)',
              u'Arts & Entertainment', u'Asian Fusion', u'Bagels',
              u'Bakeries', u'Bars', u'Breakfast & Brunch', u'Burgers',
              u'Caribbean', u'Caterers', u'Chinese', u'Coffee & Tea',
              u'Dance Clubs', u'Delis', u'Desserts', u'Diners', u'Dive Bars',
              u'Donuts', u'Event Planning & Services', u'Fast Food', u'Food',
              u'French', u'Greek', u'Grocery', u'Ice Cream & Frozen Yogurt',
              u'Indian', u'Irish', u'Italian', u'Japanese', u'Korean',
              u'Latin American', u'Lounges', u'Mediterranean', u'Mexican',
              u'Middle Eastern', u'Music Venues', u'Nightlife', u'Pizza',
              u'Pubs', u'Restaurants', u'Sandwiches', u'Seafood', u'Shopping',
              u'Spanish', u'Sports Bars', u'Steakhouses', u'Sushi Bars',
              u'Thai', u'Vietnamese']

ATTRIBUTES = {u'Wheelchair Accessible': 'wheelchair', u'Take-out': 'takeout',
              u'Alcohol': [u'beer_and_wine', u'none', u'full_bar'],
              u'Noise Level': [u'very_loud', u'average', u'loud', u'quiet'],
              u'Takes Reservations': 'reserve', u'Has TV': 'tv',
              u'Outdoor Seating': 'outdoor',
              u'Attire': [u'dressy', u'casual', u'formal'],
              u'Ambience': [u'romantic', u'intimate', u'touristy', u'hipster',
                            u'divey', u'classy', u'trendy', u'upscale',
                            u'casual'],
              u'Waiter Service': 'waiter', u'Good for Kids': 'kids',
              u'Price Range': 'price', u'BYOB': 'byob',
              u'BYOB/Corkage': [u'yes_corkage', u'yes_free', u'no'],
              u'Delivery': 'delivery',
              u'Good For': [u'dessert', u'latenight', u'lunch', u'dinner',
                            u'brunch', u'breakfast'],
              u'Parking': [u'garage', u'street', u'validated', u'lot',
                           u'valet'],
              u'Accepts Credit Cards': 'creditcard',
              u'Good For Groups': 'groups',}

def reduce_json():
    train_df = pd.read_csv('train_labels.csv.gz', compression='gzip')
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

    print(train_df.columns)
    print(id_yelp_map_df.columns)

    for fname in ('yelp_academic_dataset_business.json.gz',
                  'yelp_academic_dataset_review.json.gz',
                  'yelp_academic_dataset_user.json.gz',
                  'yelp_academic_dataset_checkin.json.gz',
                  'yelp_academic_dataset_tip.json.gz',):
        print('\n\n%s' % fname)
        with gzip.open('yelp_boston_academic_dataset/%s' % fname, 'rb')\
                as infile:
            for line in infile:
                out = json.loads(line)
                for k, v in sorted(out.items()):
                    print('%s: %s' % (k, v))
                break
    print('\n\n')

    fname = 'yelp_academic_dataset_business.json.gz'
    with gzip.open('yelp_boston_academic_dataset/%s' % fname, 'rb') as infile:
        outfile = gzip.open('business.csv.gz', 'wb')
        csv_writer = csv.writer(outfile)
        out_labels = ['restaurant_id', 'name', 'city',
                      'latitude', 'longitude', 'open', 'review_count', 
                      'avg_stars', 'neighborhoods',] + CATEGORIES
        for k, v in ATTRIBUTES.items():
            if type(v) is list:
                for _v in v:
                    out_labels.append(u'%s_%s' % (k.lower(), _v))
            else:
                out_labels.append(v)
        csv_writer.writerow(out_labels)
        for line in infile:
            out = json.loads(line)
            if out['business_id'] not in yelp_id_map:
                continue
            rid = yelp_id_map[out['business_id']]
            if rid not in id_to_ncomplaint:
                continue
            row_dict = {k: None for k in out_labels}
            row_dict['restaurant_id'] = rid
            row_dict['avg_stars'] = out['stars']
            for key in out:
                if key in out_labels:
                    val = out[key]
                    if type(val) == list:
                        if val:
                            row_dict[key] = val[0]
                    elif type(val) == unicode:
                        row_dict[key] = val.encode(errors='replace')
                    else:
                        row_dict[key] = val
            for val in CATEGORIES:
                row_dict[val] = 0
            for val in out['categories']:
                if val in out_labels:
                    row_dict[val] = 1
            for k, v in ATTRIBUTES.items():
                if type(v) == list:
                    for _k in v:
                        row_dict['%s_%s' % (k.lower(), _k)] = 0
                else:
                    row_dict[v] = 0
            for key in out['attributes'].keys():
                for k, v in ATTRIBUTES.items():
                    if k == key:
                        val = out['attributes'][key]
                        if type(v) == list and type(val) == dict:
                            for _k in val:
                                row_dict['%s_%s' % (k.lower(), _k)] = val[_k]
                        elif type(v) == list and val in v:
                            row_dict['%s_%s' % (k.lower(), val)] = 1
                        else:
                            row_dict[v] = val
            row_val = [row_dict[col] for col in out_labels]
            csv_writer.writerow(row_val)

    fname = 'yelp_academic_dataset_user.json.gz'
    with gzip.open('yelp_boston_academic_dataset/%s' % fname, 'rb') as infile:
        out_labels = [u'user_id', u'name', u'yelping_since',
                      u'votes_funny', u'votes_useful',
                      u'votes_cool', u'review_count',
                      u'friends', u'fans',  u'average_stars',
                      u'compliments_profile', u'compliments_cute',
                      u'compliments_funny', u'compliments_plain',
                      u'compliments_writer', u'compliments_list',
                      u'compliments_note', u'compliments_photos',
                      u'compliments_hot', u'compliments_more',
                      u'compliments_cool', 'elite_2005', 'elite_2006',
                      'elite_2007', 'elite_2008', 'elite_2009', 'elite_2010',
                      'elite_2011', 'elite_2012', 'elite_2014', 'elite_2015']
        outfile = gzip.open('users.csv.gz', 'wb')
        csv_writer = csv.writer(outfile)
        csv_writer.writerow(out_labels)
        for line in infile:
            out = json.loads(line)
            row_dict = {k: None for k in out_labels}
            for lab in out_labels:
                if lab == 'yelping_since':
                    year, month = [int(x) for x in out[lab].split('-')[:2]]
                    row_dict[lab] = (YEAR-year)*12 + (MONTH-month)
                elif 'votes' in lab:
                    key = lab.replace('votes_',  '')
                    if key in out['votes']:
                        row_dict[lab] = out['votes'][key]
                elif 'compliments' in lab:
                    key = lab.replace('compliments_',  '')
                    if key in out['compliments']:
                        row_dict[lab] = out['compliments'][key]
                    else:
                        row_dict[lab] = 0
                elif 'elite' in lab:
                    key = int(lab.replace('elite_', ''))
                    row_dict[lab] = int(key in out['elite'])
                elif lab == 'friends':
                    row_dict[lab] = len(out[lab])
                elif lab in out:
                    if type(out[lab]) == unicode:
                        row_dict[lab] = out[lab].encode(errors='replace')
                    else:
                        row_dict[lab] = out[lab]
            row_val = [row_dict[col] for col in out_labels]
            csv_writer.writerow(row_val)

    fname = 'yelp_academic_dataset_checkin.json.gz'
    with gzip.open('yelp_boston_academic_dataset/%s' % fname, 'rb') as infile:
        out_labels = ['restaurant_id']
        for hr_ in range(24):
            for dow in range(7):
                out_labels.append('checkin_info_%d-%d' % (hr_, dow))
        outfile = gzip.open('checkins.csv.gz', 'wb')
        csv_writer = csv.writer(outfile)
        csv_writer.writerow(out_labels)
        for line in infile:
            out = json.loads(line)
            if out['business_id'] not in yelp_id_map:
                continue
            rid = yelp_id_map[out['business_id']]
            if rid not in id_to_ncomplaint:
                continue
            row_dict = {k: None for k in out_labels}
            row_dict['restaurant_id'] = rid
            for lab in out_labels:
                if 'checkin_info' in lab:
                    key = lab.replace('checkin_info_', '')
                    if key in out['checkin_info']:
                        row_dict[lab] = out['checkin_info'][key]
                    else:
                        row_dict[lab] = 0
                elif lab in out:
                    row_dict[lab] = out[lab]
            row_val = [row_dict[col] for col in out_labels]
            csv_writer.writerow(row_val)

    fname = 'yelp_academic_dataset_tip.json.gz'
    with gzip.open('yelp_boston_academic_dataset/%s' % fname, 'rb') as infile:
        out_labels = ['user_id', 'restaurant_id', 'date']
        outfile = gzip.open('tips.csv.gz', 'wb')
        csv_writer = csv.writer(outfile)
        csv_writer.writerow(out_labels)
        for line in infile:
            out = json.loads(line)
            if out['business_id'] not in yelp_id_map:
                continue
            rid = yelp_id_map[out['business_id']]
            if rid not in id_to_ncomplaint:
                continue
            row_dict = {k: None for k in out_labels}
            row_dict['restaurant_id'] = rid
            for lab in out_labels:
                if lab in out:
                    row_dict[lab] = out[lab]
            row_val = [row_dict[col] for col in out_labels]
            csv_writer.writerow(row_val)

    fname = 'yelp_academic_dataset_review.json.gz'
    with gzip.open('yelp_boston_academic_dataset/%s' % fname, 'rb') as infile:
        ### ignore text for now
        out_labels = [u'user_id', u'review_id', u'restaurant_id',
                      u'votes_funny', u'votes_useful', u'votes_cool',
                      u'stars', u'date']
        outfile = gzip.open('reviews.csv.gz', 'wb')
        csv_writer = csv.writer(outfile)
        csv_writer.writerow(out_labels)
        for line in infile:
            out = json.loads(line)
            if out['business_id'] not in yelp_id_map:
                continue
            rid = yelp_id_map[out['business_id']]
            if rid not in id_to_ncomplaint:
                continue
            row_dict = {k: None for k in out_labels}
            row_dict['restaurant_id'] = rid
            for lab in out_labels:
                if 'votes' in lab:
                    key = lab.replace('votes_', '')
                    if key in out['votes']:
                        row_dict[lab] = out['votes'][key]
                    else:
                        row_dict[lab] = 0
                elif lab in out:
                    row_dict[lab] = out[lab]
            row_val = [row_dict[col] for col in out_labels]
            csv_writer.writerow(row_val)
    return

def feature_extraction():
    business_df = pd.read_csv('business.csv.gz', compression='gzip')
    checkin_df = pd.read_csv('checkins.csv.gz', compression='gzip')
    review_df = pd.read_csv('reviews.csv.gz', compression='gzip')
    tips_df = pd.read_csv('tips.csv.gz', compression='gzip')
    users_df = pd.read_csv('users.csv.gz', compression='gzip')

    train_df = pd.read_csv('train_labels.csv.gz', compression='gzip')
    test_df = pd.read_csv('SubmissionFormat.csv.gz', compression='gzip')

    out_labels = list(train_df.columns)
    for df in business_df, checkin_df:
        for col in df.columns:
            if col != 'restaurant_id':
                out_labels.append(col)

    out_labels += ['n_tips', u'n_review', u'votes_funny', u'votes_useful',
                   u'votes_cool', u'stars', u'w_stars', u'most_recent', 
                   u'least_recent']

    for nstar in range(1,6):
        key = 'star_%d' % nstar
        out_labels.append(key)


    business_dict = {}
    for idx, row in business_df.iterrows():
        rid = row['restaurant_id']
        row_dict = {}
        for col in out_labels:
            if col in row:
                row_dict[col] = row[col]
        if rid not in business_dict:
            business_dict[rid] = row_dict
        else:
            for col in out_labels:
                if col in row and col != 'review_count':
                    business_dict[rid][col] = max(business_dict[rid][col], 
                                                  row_dict[col])
            business_dict[rid]['review_count'] += row_dict['review_count']
    checkin_dict = {}
    for idx, row in checkin_df.iterrows():
        row_dict = {}
        for col in out_labels:
            if col in row:
                row_dict[col] = row[col]
        checkin_dict[row['restaurant_id']] = row_dict
    user_dict = {}
    for idx, row in users_df.iterrows():
        if 'average_stars' in row:
            user_dict[row['user_id']] = row['average_stars']
    review_dict = defaultdict(list)
    for idx, row in review_df.iterrows():
        row_dict = {'user_id': row['user_id']}
        for col in u'votes_funny', u'votes_useful', u'votes_cool', u'stars':
            row_dict[col] = row[col]
        for nstar in range(1,6):
            key = 'star_%d' % nstar
            if row['stars'] == nstar:
                row_dict[key] = 1
            else:
                row_dict[key] = 0
        row_dict['date'] = parse(row['date'])
        review_dict[row['restaurant_id']].append(row_dict)
    tips_dict = defaultdict(int)
    for idx, row in tips_df.iterrows():
        tips_dict[row['restaurant_id']] += 1

    for df, ofname in (train_df, 'train.csv.gz'), (test_df, 'test.csv.gz'):
        outfile = gzip.open(ofname, 'wb')
        csv_writer = csv.writer(outfile)
        csv_writer.writerow(out_labels)
    
        for idx, row in df.iterrows():
            if idx % 1000 == 0:
                print('processed %d' % idx)
            rid = row['restaurant_id']
            rdate = parse(row['date'])
            row_dict = {k: None for k in out_labels}
            for lab in out_labels:
                if lab in row:
                    row_dict[lab] = row[lab]
    
            _tmp0, _tmp1 = 2*[None]
            if rid in business_dict:
                _tmp0 = business_dict[rid]
            if rid in checkin_dict:
                _tmp1 = checkin_dict[rid]
    
            for dic in _tmp0, _tmp1:
                if not dic:
                    continue
                for lab in out_labels:
                    if lab in dic:
                        row_dict[lab] = dic[lab]
            row_dict['n_review'] = len(review_dict[rid])
            row_dict['n_tips'] = tips_dict[rid]
            row_dict['most_recent'] = 10000
            row_dict['least_recent'] = 0
            for nstar in range(1,6):
                key = 'star_%d' % nstar
                row_dict[key] = 0

            for col in (u'votes_funny', u'votes_useful', u'votes_cool', 
                        u'stars', u'w_stars'):
                row_dict[col] = 0
            for rev in review_dict[rid]:
                for col in (u'votes_funny', u'votes_useful', u'votes_cool', 
                            u'stars'):
                    row_dict[col] += rev[col]
                for nstar in range(1,6):
                    key = 'star_%d' % nstar
                    row_dict[key] += rev[key]
                datediff = abs((rdate - rev['date']).days)
                if datediff < row_dict['most_recent']:
                    row_dict['most_recent'] = datediff
                if datediff > row_dict['least_recent']:
                    row_dict['least_recent'] = datediff
                uid = rev['user_id']
                avgstar = user_dict[uid]
                row_dict['w_stars'] += (avgstar/5) * rev['stars']
    
            row_val = [row_dict[col] for col in out_labels]
            csv_writer.writerow(row_val)

    return

if __name__ == '__main__':
    reduce_json()
    feature_extraction()

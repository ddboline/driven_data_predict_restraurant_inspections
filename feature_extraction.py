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

import pandas as pd
import csv
import gzip
import json
import datetime

from dateutil.parser import parse

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
              u'Good For Groups': 'groups', u'neighborhood': 'neighborhoods'}

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
                      'latitude', 'open', 'review_count', 'stars',
                      'longitude'] + CATEGORIES
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
            for key in out:
                if key in out_labels:
                    val = out[key]
                    if type(val) == list:
                        if val:
                            row_dict[key] = val[0]
                    elif type(val) == unicode:
                        row_dict[key] = val.encode(errors='replace')
            for key in out['attributes'].keys():
                for k, v in ATTRIBUTES.items():
                    if k == key:
                        val = out['attributes'][key]
                        if type(v) == list and type(val) == dict:
                            for _k in val:
                                row_dict['%s_%s' % (k.lower(), _k)] = val[_k]
                        elif type(v) == list and val in v:
                            row_dict['%s_%s' % (k.lower(), v)] = 1
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

def feature_extraction(is_test=False):
    business_df = pd.read_csv('business.csv.gz', compression='gzip')
    checkin_df = pd.read_csv('checkins.csv.gz', compression='gzip')
    review_df = pd.read_csv('reviews.csv.gz', compression='gzip')
    tips_df = pd.read_csv('tips.csv.gz', compression='gzip')
    users_df = pd.read_csv('users.csv.gz', compression='gzip')

    if is_test:
        train_df = pd.read_csv('SubmissionFormat.csv.gz')
    else:
        train_df = pd.read_csv('train_labels.csv.gz', compression='gzip')

    out_labels = list(train_df.columns)
    for df in business_df, checkin_df:
        for col in df.columns:
            if col != 'restaurant_id':
                out_labels.append(col)

    out_labels += ['n_tips', u'n_review', u'votes_funny', u'votes_useful',
                   u'votes_cool', u'stars', u'w_stars']

    if is_test:
        outfile = gzip.open('test.csv.gz', 'wb')
    else:
        outfile = gzip.open('train.csv.gz', 'wb')
    csv_writer = csv.writer(outfile)
    csv_writer.writerow(out_labels)

    for idx, row in train_df.iterrows():
        rid = row['restaurant_id']
        row_dict = {k: None for k in out_labels}
        for lab in out_labels:
            if lab in row:
                row_dict[lab] = row[lab]

        _tmp0 = business_df[business_df['restaurant_id'] == rid]
        _tmp1 = checkin_df[checkin_df['restaurant_id'] == rid]
        _tmp2 = review_df[review_df['restaurant_id'] == rid]
        _tmp3 = tips_df[tips_df['restaurant_id'] == rid]

        for df in _tmp0, _tmp1:
            for idy, rowy in df.iterrows():
                for lab in out_labels:
                    if lab in rowy:
                        row_dict[lab] = rowy[lab]
        row_dict['n_review'] = _tmp2.shape[0]
        row_dict['n_tips'] = _tmp3.shape[0]
        for col in (u'votes_funny', u'votes_useful', u'votes_cool', u'stars',
                    u'w_stars'):
            row_dict[col] = 0

        for idy, rowy in _tmp2.iterrows():
            uid = rowy['user_id']
            star = rowy['stars']
            avgstar = float(users_df[users_df['user_id'] == uid]\
                                                ['average_stars'])
#            datediff = (parse(row['date']) - parse(rowy['date'])).days
            for col in (u'votes_funny', u'votes_useful', u'votes_cool',
                        u'stars'):
                row_dict[col] += rowy[col]
            row_dict['w_stars'] += (avgstar/5.) * star

        row_val = [row_dict[col] for col in out_labels]
        csv_writer.writerow(row_val)

    return

if __name__ == '__main__':
#    reduce_json()
    feature_extraction()

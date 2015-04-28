#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 14:12:38 2015

@author: ddboline
"""
import gzip
import json

def feature_extraction():
    with gzip.open('yelp_boston_academic_dataset/' +
                   'yelp_academic_dataset_business.json.gz', 'rb') as infile:
        for line in infile:
            out = json.loads(line)
            print out
            exit(0)
    return

if __name__ == '__main__':
    feature_extraction()

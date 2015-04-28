#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 13:48:58 2015

@author: ddboline
"""
import pandas as pd

def load_data():
    train_df = pd.read_csv('train_labels.csv.gz', compression='gzip')
    
    print train_df.columns
    return

if __name__ == '__main__':
    load_data()

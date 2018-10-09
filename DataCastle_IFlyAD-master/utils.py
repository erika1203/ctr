#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

import pandas as pd

path = '/Volumes/erika/DataCastle_IFlyAD-master/data/'

def get_data(name,  filter_=False, qcut=False, stacking=False, sample=1.0):
    name_list = list()

    if filter_:
        name_list.append("filter")
    if qcut:
        name_list.append("qcut")
    if stacking:
        name_list.append("stacking")
    if name == "train":
        name_list.append("train_features.csv")
    else:
        name_list.append("test_features.csv")

    data_name = path + '_'.join(name_list)

    df = pd.read_csv(data_name, header=0)
    if sample != 1.0:
        df = df.sample(frac=sample, random_state=2018)
    return df

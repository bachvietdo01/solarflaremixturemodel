#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 09:17:10 2021

@author: vietdo
"""

import h5py
import pandas as pd
import os
import numpy as np
from datetime import timedelta


def build_sharp_db(flare_data_path = 'HARP_with_flare_j.hdf5', out_path = 'sharp_db'):
    flare_data = h5py.File(flare_data_path, 'r')
    
    feature_list = ['USFLUX', 'MEANGAM', 'MEANGBT', 'MEANGBZ', 'MEANGBH', 'MEANJZD',
        'TOTUSJZ', 'MEANALP', 'MEANJZH', 'TOTUSJH', 'ABSNJZH', 'SAVNCPP', 'MEANPOT',
        'TOTPOT', 'MEANSHR', 'SHRGT45','SIZE', 'SIZE_ACR', 'NACR', 'NPIX']
    column_names = ["HARP_NO", "Frame", "T_REC"] + feature_list
    
    for harp in flare_data.keys():
        print('Processing ' + harp)
        sharp_df = pd.DataFrame(columns= column_names)

        for frame in flare_data[harp].keys():
            row = [harp[4:], frame]
            row += [pd.to_datetime(flare_data[harp][frame].attrs['T_REC'][:-4], format="%Y.%m.%d_%H:%M:%S")]
            row += [flare_data[harp][frame].attrs[x] for x in feature_list]
            sharp_df.loc[len(sharp_df)] = row
    
        sharp_df.to_csv(out_path + '/' + harp + '.csv', index = False)


def process_final_flare_data(time_lag = 12.0, flare_list_path = 'flare_list', sharp_db_path = 'sharp_db', 
                             out_path='flare_data'):
    
    class_mult = {'A':1e-8, 'B':1e-7, 'C':1e-6, 'M':1e-5, 'X': 1e-4}
    time_tol_in_mins = 60
    harp_to_ar = {}
    
    def convert_class(c):
        cat, score = c[0], float(c[1:])
        
        return score * class_mult[cat]
    
    for p in os.listdir(flare_list_path):    
        if p in ['.DS_Store', 'flare_list']: continue
        
        # Load GOES Flare Events
        print('Processing ' + p + '...')
        flare_events = pd.read_csv(flare_list_path + '/' + p)
        flare_events['category'] =  flare_events['class'].map(lambda x: x[0])
        flare_events['log_intensity'] = np.log(flare_events['class'].map(convert_class))
        flare_events['harp_no'] = int(p[4:-4])
        flare_events['peak_time'] = pd.to_datetime(flare_events['peak_time'])
        flare_events['ptim_lag_bf'] = flare_events['peak_time'].map(lambda t: t - timedelta(hours=time_lag))
    
        # load SHARP data
        sharp_data = pd.read_csv(sharp_db_path + '/' + p)
        sharp_data['T_REC'] = pd.to_datetime(sharp_data['T_REC'])
    
        # Merge on HARP-NO
        flares_tmp = flare_events.merge(sharp_data, how = 'left', left_on = 'harp_no', right_on = 'HARP_NO')
        flares_tmp['abs_tim_diff'] = (flares_tmp['ptim_lag_bf'] -  flares_tmp['T_REC']).map(lambda d: abs(d.total_seconds()))
        
        # add a min_tim_diff column which is the min time difference btw pt_12h_bf and T_REC group by Peak Time and Active Region
        dfg = flares_tmp.groupby(['harp_no','peak_time'])['abs_tim_diff']
        flares_tmp['min_tim_diff'] = dfg.transform(min)
    
        # pick only the row with min(min_tim_diff) and min_tim_diff < 60 minutes
        flares_data = flares_tmp.loc[flares_tmp['min_tim_diff'] == flares_tmp['abs_tim_diff']]
        harp_to_ar[p[:-4]] = set(flares_data['NOAA_ar_num'].tolist())
        flares_data = flares_data.loc[(flares_data['min_tim_diff'] <= time_tol_in_mins * 60)]
        
        flares_data.to_csv(out_path + '/' + p, index = False)
    
    return harp_to_ar

harp_to_ar = process_final_flare_data()
        
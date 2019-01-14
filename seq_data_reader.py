# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys
import tensorflow as tf
import pandas as pd
from pandas import DataFrame
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


Py3 = sys.version_info[0] == 3

tf.set_random_seed(777)

def min_max_scaling(x, min, max):
    return (x - min) / (max - min + 1e-7) # 1e-7은 0으로 나누는 오류 예방차원  

def road_data(seq_length, save_path):

    dataset = pd.read_csv('rawToRemake_seq_1days_2up_time14_output.csv', header=0)
    sort_data = dataset.sort_values(by=['date_'+str(seq_length),'product_no_'+str(seq_length)], axis=0)
    sort_data = dataset.reset_index(drop=True)

    p_l = list(sort_data.product_no_5.values)
    p_d = collections.defaultdict(lambda: 0)
    for p in p_l:
        p_d[p] += 1


    total_data = []
    te_size = 0.2

    mm_data = sort_data.copy()

    for i in range(seq_length,0,-1):
        #print('date_'+str(i))
        del mm_data['date_'+str(i)]
        del mm_data['product_no_'+str(i)]


    total_data = mm_data.values.astype(np.float)
    print('total data : ', total_data.shape)

    label_ind = total_data[0].size-1
    print('label_index : ', label_ind)
    
    split = StratifiedShuffleSplit(n_splits=1, test_size=te_size, random_state=42)
    for train_index, test_index in split.split(total_data, total_data[:,label_ind]):
        train_set = total_data[train_index]
        test_set = total_data[test_index]

    print('train data set')
    print(train_set.shape)
    print(pd.Series(train_set[:,label_ind]).value_counts())
    print('test data set')
    print(test_set.shape)
    print(pd.Series(test_set[:,label_ind]).value_counts())


    
    data_min = []
    data_max = []
    for i in range(len(total_data[0])):
        data_min.append(np.min(total_data[:,i]))
        data_max.append(np.max(total_data[:,i]))
    #print(data_min)
    #print(data_max)
    data_min = np.array(data_min)
    data_max = np.array(data_max)

    np.save(save_path + '/data_min', data_min)
    np.save(save_path + '/data_max', data_max)

    #for i in range(len(prod_data) - seq_length):
    norm_train_set = min_max_scaling(train_set, data_min, data_max)
    norm_test_set = min_max_scaling(test_set, data_min, data_max)


    #print(norm_train_set.shape[0])


    norm_train_set = norm_train_set.reshape(norm_train_set.shape[0], seq_length, -1)
    norm_test_set = norm_test_set.reshape(norm_test_set.shape[0], seq_length, -1)

    print(norm_train_set.shape)
    print(norm_test_set.shape)

    input_size = len(norm_train_set[0][0])
    print('input size : ', input_size)


    print('tr_x')
    train_x = norm_train_set[:,:,:-1]
    print(train_x.shape)

    print('tr_y')
    train_y = norm_train_set[:,:,input_size-1:]
    train_y[train_y > 0] = 1
    print(train_y.shape)


    print('te_x')
    test_x = norm_test_set[:,:,:-1]
    print(test_x.shape)

    print('te_y')
    test_y = norm_test_set[:,:,input_size-1:]
    test_y[test_y > 0] = 1
    print(test_y.shape)
    
    
    return train_x, train_y, test_x, test_y


def batch_iterator(dataX, dataY, batch_size, num_steps):

    data_len = len(dataY)
    batch_len = int(data_len / batch_size)

    #epoch_size = int((batch_len) / num_steps)
    if batch_len == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    for i in range(batch_len):
        input_x = dataX[i*batch_size: (i+1)*batch_size]
        input_y = dataY[i*batch_size: (i+1)*batch_size]
        #print(input_x.shape)
        #print(input_y.shape)
        yield (input_x, input_y)

if __name__ == "__main__":
    seq_length = 14
    save_path = 'model'
    trX, trY, teX, teY = road_data(seq_length, save_path)
    #trX, trY, vaX, vaY, teX, teY = road_data(seq_length)
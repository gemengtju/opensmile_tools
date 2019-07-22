#encoding:utf-8

from collections import defaultdict
import numpy as np
import sys
import os
from sklearn import preprocessing
import joblib


def merge_order(filepath):
    name_score = defaultdict(list)
    name_label = defaultdict(int)
    c = 0
    with open(filepath) as f:
        for line in f:
            score, label, name = line.split(' ')
            name = name.split('.wav')[0]
            name_score[name].append(float(score))
            name_label[name] = int(label)
            if int(label) == 1:
                c += 1
    print 'positive_num:', c
    count = 0
    for name in name_label:
        if name_label[name] == 1:
            count += 1
    print 'positve_name:', count
    return name_score, name_label
    

def feats_fetch_core(name_score, name_label, save_path):
    assert len(name_score) == len(name_label)
    y_labels = []
    y_scores_result = []
    y_names = []
    for name in name_score:
        # model feed
        scores = name_score[name]
        one_result = []
        for score in scores:
            one_result.append([1-score, score])
        y_scores_result.append(np.array(one_result))
        y_labels.append(name_label[name])
        y_names.append(name)
        
    x_feats = []
    for y_scores in y_scores_result:
        results = []
        results.append(np.max(y_scores, axis=0)[0])
        results.append(np.min(y_scores, axis=0)[1])
        results.append(np.min(y_scores, axis=0)[0])
        results.append(np.max(y_scores, axis=0)[1])
        # 概率均值
        results.append(np.average(y_scores, axis=0)[0])
        results.append(np.average(y_scores, axis=0)[1])
        # 分位数占比
        data_len = len(y_scores[:, 0])
        results.append(float(len(np.where(y_scores[:, 0] > 0.9))) / data_len)
        results.append(float(len(np.where(y_scores[:, 1] > 0.9))) / data_len)
        results.append(float(len(np.where(y_scores[:, 0] > 0.8))) / data_len)
        results.append(float(len(np.where(y_scores[:, 1] > 0.8))) / data_len)
        results.append(float(len(np.where(y_scores[:, 0] > 0.7))) / data_len)
        results.append(float(len(np.where(y_scores[:, 1] > 0.7))) / data_len)
        results.append(float(len(np.where(y_scores[:, 0] > 0.6))) / data_len)
        results.append(float(len(np.where(y_scores[:, 1] > 0.6))) / data_len)
        results.append(float(len(np.where(y_scores[:, 0] > 0.5))) / data_len)
        results.append(float(len(np.where(y_scores[:, 1] > 0.5))) / data_len)
        results.append(float(len(np.where(y_scores[:, 0] > 0.4))) / data_len)
        results.append(float(len(np.where(y_scores[:, 1] > 0.4))) / data_len)
        results.append(float(len(np.where(y_scores[:, 0] > 0.3))) / data_len)
        results.append(float(len(np.where(y_scores[:, 1] > 0.3))) / data_len)
        results.append(float(len(np.where(y_scores[:, 0] > 0.2))) / data_len)
        results.append(float(len(np.where(y_scores[:, 1] > 0.2))) / data_len)
        results.append(float(len(np.where(y_scores[:, 0] > 0.1))) / data_len)
        results.append(float(len(np.where(y_scores[:, 1] > 0.1))) / data_len)
        x_feats.append(results)

    x_feats = np.array(x_feats)
    if os.path.exists('./workspace/package_feature/train_scaler_20181105_second.m'):
        scaler = joblib.load('./workspace/package_feature/train_scaler_20181105_second.m')
    else:
        scaler = preprocessing.StandardScaler().fit(x_feats)
        joblib.dump(scaler, './workspace/package_feature/train_scaler_20181105_second.m')

    x_feats = scaler.transform(x_feats)
    y_labels = np.array(y_labels)
    y_names = np.array(y_names)
    np.savez(save_path, x=x_feats, y=y_labels, name=y_names)


if __name__ == '__main__':
    name_score, name_label = merge_order(filepath='./workspace/exp/top_result_20181105_test_second.txt')
    feats_fetch_core(name_score, name_label, './workspace/package_feature/train_test_data/20181105_test_segment0')
    

# encoding:utf-8


import joblib
import numpy as np
import json
import os
from concurrent.futures import ProcessPoolExecutor
import random
from sklearn import preprocessing

# IF path is not exist, create the path
def makedir(dirs):
    if not os.path.exists(dirs):
        os.makedirs(dirs)


def fetch_filelist(root_path, label=0):
    labelpathlist = os.listdir(root_path)
    labelpathlist = [root_path + '/' + x for x in labelpathlist]
    labels = [label] * len(labelpathlist)
    return labelpathlist, labels


def fetch_pathlist(root_path):
    labelpathlist = os.listdir(root_path)
    wholepaths = []
    wholelabels = []
    for labelpath in labelpathlist:
        if labelpath == 'ch1':
            label = 0
        else:
            label = 0
        path = root_path + '/' + labelpath
        files = os.listdir(path)
        labels = [label] * len(files)
        wholepaths.extend([path + '/' + x for x in files])
        wholelabels.extend(labels)
    return wholepaths, wholelabels


def data_gen(pathlabels, threads):
    length = len(pathlabels) / threads
    i = 0
    index = 0
    while i < threads:
        if i == threads - 1:
            i += 1
            yield pathlabels[index:]
        else:
            i += 1
            result = pathlabels[index: index + length]
            index += length
            yield result
    yield None


def pre_feats_fetch(label, threads=1, root_path='data_mock/', stride=10, save_path='biaozhu_drunk'):
    makedir(save_path)
    executor = ProcessPoolExecutor()
    futures = []
    pathlabels = zip(*fetch_filelist(root_path, label))
    random.shuffle(pathlabels)
    subdata_gen = data_gen(pathlabels, threads)
    subdata = subdata_gen.next()
    distance = stride / 10
    i = 0
    while subdata:
        futures.append(executor.submit(pre_feats_fetch_core, (zip(*subdata), i, distance, save_path)))
        i += 1
        subdata = subdata_gen.next()
    for x in futures:
        x.result()


def pre_feats_fetch_core((pathlabellist, index, distance, save_path)):
    pathlist, labellist = pathlabellist
    assert len(pathlist) == len(labellist)
    results = []
    names = []
    splits = 2
    size = len(pathlist) / splits
    split_count = 0
    split_index = 0
    for i, path in enumerate(pathlist):
        with open(path) as f:
            for line in f:
                if not line or "@" in line:
                    continue
                if not line.strip('\n'):
                    continue

                feats = line.split(',')[1: -1]
                if len(feats) != 384:
                    continue
                if i % distance != 0:
                    continue
                feats.append(labellist[i])
                results.append(feats)
                names.append(line.split(',')[0].split('/')[-1])
        print path, size, line.split(',')[0].split('/')[-1]
        
        split_count += 1
        if split_count >= size:
            np.savez(save_path + '/train_%s_%s' % (str(index), str(split_index)), x=np.array(results)[:, :-1].astype(np.float32), y=np.array(results)[:, -1].astype(np.int32), name=np.array(names))
            split_index += 1
            split_count = 0
            results = []
            names = []


def numpy_test(root_path):
    pathlist = os.listdir(root_path)
    pathlist = [root_path + '/' + x for x in pathlist if 'train_' in x]
    lengths = 0
    for i, path in enumerate(pathlist):
        data = np.load(path)
        print path, len(data['x']), len(data['y']), len(data['name'])
        assert len(data['x']) == len(data['y']) == len(data['name'])
        lengths += len(data['x'])
    print 'size: ', lengths


def feats_data_test(root_path):
    filepaths, filelabels = fetch_filelist(root_path)
    assert len(filepaths) == len(filelabels)
    lengths = 0
    for i, path in enumerate(filepaths):
        with open(path) as f:
            for line in f:
                if not line or "@" in line:
                    continue
                if not line.strip('\n'):
                    continue

                feats = line.split(',')[1: -1]
                if len(feats) != 384:
                    continue
                lengths += 1
    print 'size: ', lengths


# assemble
def assemble(root_path, prefix='qx2w_'):
    paths = os.listdir(root_path)
    paths = [root_path + '/' + x for x in paths if 'train_' in x]
    x = []
    y = []
    name = []
    for path in paths:
        loaded = np.load(path)
        x.append(loaded['x'])
        y.append(loaded['y'])
        name.append(loaded['name'])
    x = np.concatenate(x, axis=0)
    y = np.concatenate(y, axis=0)
    name = np.concatenate(name, axis=0)
    assert len(x) == len(y) == len(name)
    data = zip(x, y, name)
    np.random.shuffle(data)
    if data:
        x, y, names = zip(*data)
        np.savez(root_path + '/' + prefix, x=x, y=y, name=names)


# train first
def merge_and_shuffle(paths, prefix='train_', splits=4):
    x = []
    y = []
    name = []
    for path in paths:
        loaded = np.load(path)
        x.append(loaded['x'])
        y.append(loaded['y'])
        print 'loaded:', path, len(loaded['x'])
        name.append(loaded['name'])
    x = np.concatenate(x, axis=0)
    y = np.concatenate(y, axis=0)
    name = np.concatenate(name, axis=0)
    assert len(x) == len(y) == len(name)
    if os.path.exists('./workspace/package_feature/train_scaler_20181105.m'):
        scaler = joblib.load('./workspace/package_feature/train_scaler_20181105.m')
    else:
        scaler = preprocessing.StandardScaler().fit(x)
        joblib.dump(scaler, './workspace/package_feature/train_scaler_20181105.m')
    x = scaler.transform(x)
    data = zip(x, y, name)
    np.random.shuffle(data)
    size = len(data) / splits
    for i in range(splits):
        start = i * size
        end = (i+1) * size if i != splits - 1 else len(data)
        x, y, names = zip(*data[start: end])
        print 'data:', len(x)
        makedir(prefix[:prefix.rfind("/")])
        np.savez(prefix + str(i), x=x, y=y, name=names)


if __name__ == '__main__':
    paths = [('/data1/gemeng/feature/feature_20181105/train/drunk_vad', './workspace/package_feature/20181105_chengke_segment_npy/train/drunk/', 1),
             ('/data1/gemeng/feature/feature_20181105/train/no_drunk_vad', './workspace/package_feature/20181105_chengke_segment_npy/train/no_drunk/', 0),
             ('/data1/gemeng/feature/feature_20181105/test/drunk_vad', './workspace/package_feature/20181105_chengke_segment_npy/test/drunk/', 1),
             ('/data1/gemeng/feature/feature_20181105/test/no_drunk_vad', './workspace/package_feature/20181105_chengke_segment_npy/test/no_drunk/', 0),
             ('/data1/gemeng/feature/feature_20181105/wav_train/drunk_vad', './workspace/package_feature/20181105_chengke_segment_npy/wav_train/drunk/', 1),
             ('/data1/gemeng/feature/feature_20181105/wav_train/no_drunk_vad', './workspace/package_feature/20181105_chengke_segment_npy/wav_train/no_drunk/', 0)]
    '''
    paths = [('/data1/gemeng/feature/feature_20181101/test/drunk_vad', 'repaired_new_vad_segment_npy/test/drunk/', 1),
             ('/data1/gemeng/feature/feature_20181101/test/no_drunk_vad', 'repaired_new_vad_segment_npy/test/no_drunk/', 0),
             ('/data1/gemeng/feature/feature_20181101/train/drunk_vad', 'repaired_new_vad_segment_npy/train/drunk/', 1),
             ('/data1/gemeng/feature/feature_20181101/train/no_drunk_vad', 'repaired_new_vad_segment_npy/train/no_drunk/', 0),
             ('/data1/gemeng/feature/feature_20181101/wav_train/drunk_vad', 'repaired_new_vad_segment_npy/wav_train/drunk/', 1),
             ('/data1/gemeng/feature/feature_20181101/wav_train/no_drunk_vad', 'repaired_new_vad_segment_npy/wav_train/no_drunk/', 0)]
    paths = [('/data1/zhangruixiong/drunk_data/new_train_feature/drunk_vad', '20181105_old_segment_npy/train/drunk/', 1),
             ('/data1/zhangruixiong/drunk_data/new_train_feature/no_drunk_vad', '20181105_old_segment_npy/train/no_drunk/', 0),
             ('/data1/gemeng/feature/feature_20181105/test/drunk_vad', '20181105_old_segment_npy/test/drunk/', 1),
             ('/data1/gemeng/feature/feature_20181105/test/no_drunk_vad', '20181105_old_segment_npy/test/no_drunk/', 0),
             ('/data1/zhangruixiong/drunk_data/new_wav_train_feature/drunk_vad', '20181105_old_segment_npy/wav_train/drunk/', 1),
             ('/data1/zhangruixiong/drunk_data/new_wav_train_feature/no_drunk_vad', '20181105_old_segment_npy/wav_train/no_drunk/', 0)]
    '''
   # for feat_path, save_path, label in paths:
   #     #feats_data_test(root_path=feat_path)
   #     pre_feats_fetch(label=label, threads=10, root_path=feat_path, stride=10, save_path=save_path)
   #     assemble(save_path, prefix='assemble')
   #     #numpy_test(root_path=save_path)
    '''
    merge_and_shuffle(['20181105_old_segment_npy/train/drunk/assemble.npz', '20181105_old_segment_npy/train/no_drunk/assemble.npz'], prefix='train_test_data/20181105_old_train_segment', splits=1)
    merge_and_shuffle(['20181105_old_segment_npy/test/drunk/assemble.npz', '20181105_old_segment_npy/test/no_drunk/assemble.npz'], prefix='train_test_data/20181105_old_test_segment', splits=1)
    merge_and_shuffle(['20181105_old_segment_npy/wav_train/drunk/assemble.npz', '20181105_old_segment_npy/wav_train/no_drunk/assemble.npz'], prefix='train_test_data/20181105_old_wav_train_segment', splits=1)
    '''
    merge_and_shuffle(['./workspace/package_feature/20181105_chengke_segment_npy/train/drunk/assemble.npz', './workspace/package_feature/20181105_chengke_segment_npy/train/no_drunk/assemble.npz'], prefix='./workspace/package_feature/train_test_data/20181105_train_segment', splits=1)
    merge_and_shuffle(['./workspace/package_feature/20181105_chengke_segment_npy/test/drunk/assemble.npz', './workspace/package_feature/20181105_chengke_segment_npy/test/no_drunk/assemble.npz'], prefix='./workspace/package_feature/train_test_data/20181105_test_segment', splits=1)
    merge_and_shuffle(['./workspace/package_feature/20181105_chengke_segment_npy/wav_train/drunk/assemble.npz', './workspace/package_feature/20181105_chengke_segment_npy/wav_train/no_drunk/assemble.npz'], prefix='./workspace/package_feature/train_test_data/20181105_wav_train_segment', splits=1)

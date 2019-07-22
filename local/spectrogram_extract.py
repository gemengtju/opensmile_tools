#!/bin/python

"""
Summary:  Prepare data. 
Author:   Meng Ge
Created:  2018.11.15
"""

import os
import soundfile
import numpy as np
import argparse
import csv
import time
#import matplotlib.pyplot as plt
from scipy import signal
import pickle
import cPickle
import h5py
from sklearn import preprocessing
#import prepare_data as pp_data
import config as cfg


def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)

def read_audio(path, target_fs=None):
    (audio, fs) = soundfile.read(path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if target_fs is not None and fs != target_fs:
        audio = librosa.resample(audio, orig_sr=fs, target_sr=target_fs)
        fs = target_fs
    return audio, fs

def pad_with_border(x, n_pad):

    """
    Pad the begin and finish of spectrogram with border frame value. 
    """
    x_pad_list = [x[0:1]] * n_pad + [x] + [x[-1:]] * n_pad
    return np.concatenate(x_pad_list, axis=0)

def log_sp(x):
    return np.log(x + 1e-08)

def mat_2d_to_3d(x, agg_num, hop):
    """
    Segment 2D array to 3D segments. 
    """
    # Pad to at least one block. 
    len_x, n_in = x.shape
    if (len_x < agg_num):
        x = np.concatenate((x, np.zeros((agg_num - len_x, n_in))))
        
    # Segment 2d to 3d. 
    len_x = len(x)
    i1 = 0
    x3d = []
    while (i1 + agg_num <= len_x):
        x3d.append(x[i1 : i1 + agg_num])
        i1 += hop
    return np.array(x3d)

def calc_sp(audio, mode):
    """
    Calculate spectrogram. 
    
    Args:
      audio: 1darray. 
      mode: string, 'magnitude' | 'complex'
    Returns:
      spectrogram: 2darray, (n_time, n_freq). 
    """
    n_window = cfg.n_window
    n_overlap = cfg.n_overlap
    ham_win = np.hamming(n_window)
    [f, t, x] = signal.spectral.spectrogram(
                       audio,
                       window=ham_win,
                       nperseg=n_window,
                       noverlap=n_overlap,
                       detrend=False,
                       return_onesided=True,
                       mode=mode)
    x = x.T
    if mode == 'magnitude':
        x = x.astype(np.float32)
    elif mode == 'complex':
        x = x.astype(np.complex64)
    else:
        raise Exception("Incorrect model")
    return x

def calc_speech_features(args):
    """
    Calculate spectrogram features for audios. Then write the features to disk.

    Args:
      workspace: str, path of workspace. 
      speech_dir: str, path of speech data. 
      noise_dir: str, path of noise data. 
      data_type: str, 'train' | 'test'. 
      snr: float, signal to noise ratio to be mixed. 
    """
    print(args)
    workspace = args.workspace
    #speech_path = args.speech_path
    data_type = args.data_type
    fs = cfg.sample_rate
    
    speech_path = os.path.join("./lists", "%s.lst" % data_type)
    with open(speech_path, 'rb') as f:
        speech_list = f.readlines()

    for speech_item in speech_list:
        speech_name = speech_item.split("/")[-1]
        (speech_audio, _) = read_audio(speech_item.replace("\n", ""), target_fs=fs)
        speech_complex_x = calc_sp(speech_audio, mode='complex')
        if speech_item.split("/")[-2] == "drunk":
            speech_y = 1
        else:
            speech_y = 0
        out_feat_path = os.path.join(workspace, "feature", "spectrogram", data_type, "%s.p" % speech_name.replace("\n", ""))
        create_folder(os.path.dirname(out_feat_path))
        data = [speech_name.replace("\n", ""), speech_complex_x, speech_y]
        cPickle.dump(data, open(out_feat_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)

def pack_features(args):
    print(args)
    workspace = args.workspace
    data_type = args.data_type
    n_concat = args.n_concat
    n_hop = args.n_hop

    cnt = 0
    x_all = []
    y_all = []

    feat_dir = os.path.join(workspace, "feature", "spectrogram", data_type)
    names = os.listdir(feat_dir)
    for na in names:
        feat_path = os.path.join(feat_dir, na)
        data = cPickle.load(open(feat_path, 'rb'))
        [speech_name, speech_complex_x, speech_y] = data
        print(speech_name)
        print(speech_y)
        speech_x = np.abs(speech_complex_x)

        #n_pad = (n_concat -1) / 2
        #speech_x = pad_with_border(speech_mag_x, n_pad)
        speech_x_3d = mat_2d_to_3d(speech_x, agg_num=n_concat, hop=n_hop)
        x_all.append(speech_x_3d)
        y_all.append(np.array([[speech_y]]))

        if cnt % 100 == 0:
            print(cnt)
        cnt += 1

    print(y_all)
    x_all = np.concatenate(x_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)

    x_all = log_sp(x_all).astype(np.float32)

    out_path = os.path.join(workspace, "package_feature", "spectrogram", data_type, "data.h5")
    create_folder(os.path.dirname(out_path))
    with h5py.File(out_path, 'w') as hf:
        hf.create_dataset('x', data=x_all)
        hf.create_dataset('y', data=y_all)

    print("Pack features finished!")
    

def compute_scaler(args):
    print(args)
    workspace = args.workspace
    data_type = args.data_type

    hdf5_path = os.path.join(workspace, "package_feature", "spectrogram", data_type, "data.h5")
    with h5py.File(hdf5_path, 'r') as hf:
        x = hf.get('x')
        x = np.array(x)

    (n_segs, n_concat, n_freq) = x.shape
    x2d = x.reshape((n_segs * n_concat, n_freq))
    scaler = preprocessing.StandardScaler(with_mean=True, with_std=True).fit(x2d)
    print(scaler.mean_)
    print(scaler.scale_)

    out_path = os.path.join(workspace, "package_feature", "spectrogram", data_type, "scaler.p")
    create_folder(os.path.dirname(out_path))
    pickle.dump(scaler, open(out_path, 'wb'))
    print("Compute scaler finished!")





         

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')

    parser_calc_speech_features = subparsers.add_parser('calc_speech_features')
    parser_calc_speech_features.add_argument('--workspace', type=str, required=True)
    parser_calc_speech_features.add_argument('--data_type', type=str, required=True)
    

    parser_package_features = subparsers.add_parser('package_features')
    parser_package_features.add_argument('--workspace', type=str, required=True)
    parser_package_features.add_argument('--data_type', type=str, required=True)
    parser_package_features.add_argument('--n_concat', type=int, required=True)
    parser_package_features.add_argument('--n_hop', type=int, required=True)

    parser_compute_scaler = subparsers.add_parser('compute_scaler')
    parser_compute_scaler.add_argument('--workspace', type=str, required=True)
    parser_compute_scaler.add_argument('--data_type', type=str, required=True)

    args = parser.parse_args()
    if args.mode == 'calc_speech_features':
        calc_speech_features(args)
    elif args.mode == 'package_features':
        pack_features(args)
    elif args.mode == 'compute_scaler':
        compute_scaler(args)
    else:
        raise Exception("Error!")
    print("start extract spectrogram...")





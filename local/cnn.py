#!/bin/python
"""
Summary:  CNN Model. 
Author:   Meng Ge
Created:  2018.11.16
"""
import argparse

def load_hdf5(hdf5_path):
    """Load hdf5 data. 
    """
    with h5py.File(hdf5_path, 'r') as hf:
        x = hf.get('x')
        y = hf.get('y')
        x = np.array(x)     # (n_segs, n_concat, n_freq)
        y = np.array(y)     # 1: drunk, 0: no_drunk        
    return x, y

def train(args):
    print(args)
    workspace = args.workspace
    lr = args.lr

    tr_hdf5_path = os.path.join(workspace, "package_feature", "spectrogram", "wav_train", "data.h5")
    te_hdf5_path = os.path.join(workspace, "package_feature", "spectrogram", "test", "data.h5")
    (tr_x, tr_y) = pp_data.load_hdf5(tr_hdf5_path)
    (te_x, te_y) = pp_data.load_hdf5(te_hdf5_path)
    print(tr_x.shape, tr_y.shape)
    print(te_x.shape, te_y.shape)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--workspace', type=str, required=True)
    parser_train.add_argument('--lr', type=float, required=True)

    args = parser.parse_args()

    if args.mode == 'train':
        train(args)
    else:
        raise Exception("Error!")

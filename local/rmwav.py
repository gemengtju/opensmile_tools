#!/bin/python

import os
# list the paths of all audios from a file containing each audio paths
def list_audio_path_from_file(file_path):
    L = []
    with open(file_path, "r") as f:
            L = f.readlines()
    return L

if __name__ == "__main__":
    lst = list_audio_path_from_file("/data1/zhangruixiong/drunk_data/yuyi_del/wav_train_arff.txt")
    for item in lst:
        print(item)
        os.system("rm %s" % item)

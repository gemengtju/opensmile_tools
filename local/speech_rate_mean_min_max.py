#!/bin/python

import numpy as np

with open("./speech_rate_guilin.txt", "r") as f:
    lines = f.readlines()

fw = open("./feature/speechrate/speechrate_mean_min_max_feats.txt", "w")
dur_list = []
init_wav_name = lines[0].split(" ")[0].split("/")[-1].split(".")[0]
for idx in range(len(lines)):
    items = lines[idx].split(" ")
    name = items[0]
    duration = float(items[1])
    temp_wav_name = name.split("/")[-1].split(".")[0]
    if temp_wav_name == init_wav_name and idx != len(lines)-1:
        dur_list.append(duration)
    else:
        print("%s.wav %s %s %s" % (temp_wav_name, np.mean(dur_list), np.min(dur_list), np.max(dur_list)))
        fw.write("%s.wav %s %s %s\n" % (temp_wav_name, np.mean(dur_list), np.min(dur_list), np.max(dur_list)))
        dur_list = []
        dur_list.append(duration)
        init_wav_name = temp_wav_name

fw.close()    
    
    
    

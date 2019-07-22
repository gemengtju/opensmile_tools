#!/bin/python
from get_frequency import *

##### Time ######
#with open("./data/results.list") as f:
#    lines = f.readlines()
#all_order_dict = {}
#for line in lines:
#    all_order_dict[line.split(",")[0].replace("\n","")] = line.split(",")[3].split(" ")[1].split(":")[0].replace("\n","")
#
with open("/data1/zhangruixiong/feature_platform/positive_id.list") as ff:
    drunk_lines = ff.readlines()
#print(drunk_lines)
#
#drunk_time_count = [0]*24
#for drunk_line in drunk_lines:
#    time_val = all_order_dict[drunk_line.replace("\n","")]
#    drunk_time_count[int(time_val)] = drunk_time_count[int(time_val)] + 1
#print(drunk_time_count)


######## Province, Distance, Age #########
with open("/data1/zhangruixiong/feature_platform/result.list") as f_ext:
    ext_lines = f_ext.readlines()

drunkorder_province_list = []
drunkorder_age_list = []
for ext_line in ext_lines:
    str = ext_line.split(",")[0]+"\n"
    if str in drunk_lines:
        drunkorder_province_list.append(ext_line.split(",")[1].replace("\n",""))
        drunkorder_age_list.append(ext_line.split(",")[2].replace("\n",""))
print(all_list(drunkorder_province_list).keys())
print(all_list(drunkorder_age_list).keys())
print(all_list(drunkorder_age_list).values())
print(all_list(drunkorder_province_list).values())


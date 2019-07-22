#!/bin/bash

#awk '{print $1}' $1 > ./wav_name.txt
#awk '{print $2}' $1 > ./wav_text.txt
#awk '{print length($0)}' ./wav_text.txt > zishu.txt
#paste -d' ' ./wav_name.txt zishu.txt > ./wav_name_zishu.txt
#
#
##for line in `cat ./ddwav_name_zishu.txt`
#cat ./wav_name_zishu.txt | while read line
#do
#    #name=`awk '{print $1}' $line`
#    #echo $line
#    name=`echo $line|awk -F ' ' '{print $1}'`
#    zishu=`echo $line|awk -F ' ' '{print $2}'`
#    #echo $zishu
#    #echo $zishu
#    #echo $zishu >> tmp.txt
#    prefix=${name%.*}
#    tm=`sox ${prefix}* -n stat 2>&1 | sed -n 's#^Length (seconds):[^0-9]*\([0-9.]*\)$#\1#p'`
#    rate=`echo "scale=2; $zishu/$tm" | bc`
#    echo ${name} ${zishu} ${tm} ${rate}
#    echo ${name} ${rate} >> speech_rate_guilin.txt
#    #echo $tm
#done


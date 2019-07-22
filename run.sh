#!/bin/bash

# usage: copy this sample file to your working dir and execute
# qsub sample_script_serial.sh
# after modifying necessary options.
# requesting options of sge start with #$
# feel free to modify any below to fulfill task requirement

#$ -N siyuke 
#$ -cwd

# merge stdo and stde to one file
#$ -j y

# preserving your environment if necessary
#$ -V

# resource requesting, e.g. for gpu use 
#$ -l gpu=1

# start whatever your job below, e.g., python, matlab, etc.

# /opt18/matlab_2015b/bin/matlab -r scriptTrainDNN_cIRM
sh drunk_main_sicheng.sh

#/opt18/matlab_2015b/bin/matlab -r simdata_process
#/opt18/matlab_2015b/bin/matlab -r get_score

hostname;
echo 'Done.'

#!/bin/bash
# by gemeng 20181018

### VAD parameters
VAD_MODE="wav_path"
INFILE_PATH="/data1/gemeng/data/sicheng_1w_tousu_3k_labeled/no_drunk_tmp.list"
OUTPUT_DIR="/data1/gemeng/data/sicheng_1w_tousu_3k_labeled/sicheng_3k_labeled_vad/no_drunk/"
#TR_DATA_DIR="/data1/gemeng/drunk_asr/data/sicheng/mp3_lr_wav/"

### VAD DIDI version parameters (ps: output dir is current dir of wav path)
VAD_DIDI_DIR="/data1/gemeng/tools/vadtools_didi/modelVad_8k/"
INFILE_PATH="/data1/gemeng/tools/vadtools_didi/modelVad_8k/demo_wavs/test_list"

### After VAD, pcm --> wav
PCM_WAVS_DIR="/data1/gemeng/tools/vadtools_didi/modelVad_8k/demo_wavs"
PCM_WAV_STR_PATTERN=".wav"

####================================= siyuke must care about it so important=================================
### OPENSmile feature extraction function parameters
OUTPUT_DIR_AFTER_PCM2WAV="./data/"
AUDIO_STR_PATTERN=".wav"
OPENSMILE_DIR="./tools/opensmile-2.3.0"
TR_VAD_DATA_DIR=$OUTPUT_DIR_AFTER_PCM2WAV
OUTPUT_OPENSMILE_BAT_CONF_NAME="tmp_conf"
OPENSMILE_FEATURE_OUTPUT_PATH="./feature/opensmile_feature/"
OUTPUT_OPENSMILE_BAT_CONF_PATH=${OPENSMILE_DIR}/${OUTPUT_OPENSMILE_BAT_CONF_NAME}
JOBS_NUM=2 # the max number of the cpus
####==================================================================


# Speech rate extraction parameters
TRANS_FILE_PATH="/data1/zhangruixiong/drunk_asr/workspace/semantic/recognition.utf.list"

# Speech spetrogram extraction parameters
WORKSPACE="./workspace"


# Stage
stage=10

if [ $stage -le 10 ]; then
    #echo "VAD DIDI Version start"
    #python prepare_data.py vad_didi --vad_didi_dir=$VAD_DIDI_DIR --infile_path=$INFILE_PATH
    #echo "VAD DIDI end"
    #
    #echo "PCM2WAV start"
    #python prepare_data.py pcm2wav --pcm_wavs_dir=$PCM_WAVS_DIR --pcm_wav_str_pattern=$PCM_WAV_STR_PATTERN --output_dir_after_pcm2wav=$OUTPUT_DIR_AFTER_PCM2WAV
    #echo "PCM2WAV end"
    
    #echo "VAD start."
    #python prepare_data.py vad --vad_mode=$VAD_MODE --infile_path=$INFILE_PATH --output_dir=$OUTPUT_DIR
    #echo "VAD end."
    
    echo "create opensmile bat conf file."
    python ./local/prepare_data.py create_opensmile_bat_conf --opensmile_dir=$OPENSMILE_DIR --output_opensmile_bat_conf_name=$OUTPUT_OPENSMILE_BAT_CONF_NAME --tr_vad_data_dir=$TR_VAD_DATA_DIR --audio_str_pattern=$AUDIO_STR_PATTERN --opensmile_feature_output_path=$OPENSMILE_FEATURE_OUTPUT_PATH
    chmod +x $OUTPUT_OPENSMILE_BAT_CONF_PATH
    echo "opensmile bat conf created."
    
    
    #echo "Extract the opensmile feature."
    #sh $OUTPUT_OPENSMILE_BAT_CONF_PATH
    #cho "feature files finished"
    
    echo "Extract the opensmile feature via batch operation"
    ALL_TASK_NUM=`cat $OUTPUT_OPENSMILE_BAT_CONF_PATH | wc -l`
    TASK_NUM_EACH_JOB=`expr $ALL_TASK_NUM / $JOBS_NUM + 1`
    split -l $TASK_NUM_EACH_JOB $OUTPUT_OPENSMILE_BAT_CONF_PATH -d ${OUTPUT_OPENSMILE_BAT_CONF_PATH}_sp
    chmod +x ${OUTPUT_OPENSMILE_BAT_CONF_PATH}_sp*
    for((idx=0;idx<$JOBS_NUM;idx++));do
        idx_tmp=`printf "%02d\n" $idx`
        sh ${OUTPUT_OPENSMILE_BAT_CONF_PATH}_sp${idx_tmp} &
    done
    echo "feature batch extracted finished!"
fi

if [ $stage -le 2 ]; then
    #python ./model/model_fetch_data.py
    #python ./model/main.py
    python ./model/merge_result.py
fi

# Extract speech rate
if [ $stage -le 3 ]; then
    #sh ./local/speech_rate_extract.sh $TRANS_FILE_PATH
    python ./local/speech_rate_mean_min_max.py
fi

# Extract some extractal factors of orders
if [ $stage -le 5 ]; then
    # Time, Province, Distance, Age
    python ./local/drunk_extractal_feats.py
fi

if [ $stage -le 4 ]; then
    #python ./local/spectrogram_extract.py calc_speech_features --workspace=$WORKSPACE --data_type="wav_train"
    #python ./local/spectrogram_extract.py calc_speech_features --workspace=$WORKSPACE --data_type="test"
    python ./local/spectrogram_extract.py package_features --workspace=$WORKSPACE --data_type="test" --n_concat="11" --n_hop="3"
    python ./local/spectrogram_extract.py package_features --workspace=$WORKSPACE --data_type="wav_train" --n_concat="11" --n_hop="3"
    python ./local/spectrogram_extract.py compute_scaler --workspace=$WORKSPACE --data_type="wav_train"
    python ./local/spectrogram_extract.py compute_scaler --workspace=$WORKSPACE --data_type="test"
    #python ./local/cnn.py train --workspace=$WORKSPACE --lr="0.1"
fi


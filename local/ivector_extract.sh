#!/bin/bash
#by Damon 20181101
'''
This code is based on sre10 example of KALDI tools
You need download KALDI tools before you use this code.
'''

data_dir="/home/luban/gemeng/speech_seperation/tools/kaldi/egs/sre10/v1/wavs"
mfccdir="/home/luban/gemeng/speech_seperation/tools/kaldi/egs/sre10/v1/mfcc"
vaddir="/home/luban/gemeng/speech_seperation/tools/kaldi/egs/sre10/v1/mfcc"
stage=15

echo "Start..."

# prepare wav.scp, utt2spk, spk2utt
if [ $stage -le 10 ]; then
    for x in wav_train wav_test; do
        mkdir -p data/$x
        find $data_dir/$x -iname "*.wav" > data/$x/wav.scp.path
        cat data/$x/wav.scp.path | awk -F '/' '{printf("%s\n", $NF)}' | sed 's|.wav||' > data/$x/wav.scp.id
        paste -d' ' data/$x/wav.scp.id data/$x/wav.scp.path > data/$x/wav.scp
        
        cat data/$x/wav.scp.id | awk -F '.' '{printf("%s\n", $(NF-1))}' > data/$x/speaker_id
        paste -d' ' data/$x/wav.scp.id data/$x/speaker_id > data/$x/utt2spk
        
        utils/utt2spk_to_spk2utt.pl data/$x/utt2spk > data/$x/spk2utt
        
        rm data/$x/wav.scp.id
        rm data/$x/wav.scp.path
        rm data/$x/speaker_id
    done
fi



# MFCC extraction
if [ $stage -le 11 ]; then
    for x in wav_train wav_test; do
        utils/fix_data_dir.sh data/$x
        steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 10 --cmd "run.pl" \
            data/$x exp/make_mfcc $mfccdir
    done
fi

# VAD
if [ $stage -le 12 ]; then
    for x in wav_train wav_test; do
        sid/compute_vad_decision.sh --nj 10 --cmd "run.pl" \
            data/$x exp/make_vad $vaddir
    done
fi

# merge operation
if [ $stage -le 13 ]; then
    for x in wav_train wav_test; do
        cat mfcc/raw_mfcc_$x*.scp > data/$x/feats.scp
        cat mfcc/vad_$x*.scp > data/$x/vad.scp
    done
fi

# Train UBM and i-vector extractor using traning data
if [ $stage -le 14 ]; then
    utils/fix_data_dir.sh data/wav_train/
    sid/train_diag_ubm.sh --cmd "run.pl" \
        --nj 10 --num-threads 8 \
        data/wav_train 2048 \
        exp/diag_ubm_2048

    sid/train_full_ubm.sh --nj 10 --remove-low-count-gaussians false \
        --cmd "run.pl" data/wav_train \
        exp/diag_ubm_2048 exp/full_ubm_2048

    sid/train_ivector_extractor.sh --num_iters=5 exp/full_ubm_2048/final.ubm data/wav_train exp/extractor
fi


# Extract i-vectors
if [ $stage -le 15 ]; then
    for x in wav_train wav_test; do
        sid/extract_ivectors.sh --cmd "run.pl" --nj 10 \
            exp/extractor data/$x \
            exp/ivectors_$x
    done
fi

echo "End..."

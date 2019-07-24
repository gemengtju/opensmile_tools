#/usr/bin/python
# -*- coding: UTF-8 -*-
# by Meng Ge 20181013

import os
import wave
import argparse

# list the paths of all audios
def list_audio_path(file_dir, str_pattern):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == str_pattern:
                L.append(os.path.join(root, file))
    return L

# list the paths of all audios using second string
def list_audio_path_2(file_dir, str_pattern):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(os.path.splitext(file)[0])[1] == str_pattern:
                L.append(os.path.join(root, file))
    return L

# list the paths of all audios from a file containing each audio paths
def list_autio_path_from_file(file_path):
    L = []
    with open(file_path, "r") as f:
	    L = f.readlines()
    return L

# get the duration of an audio
def get_audio_duration(audio_path):
    audiofile = wave.open(audio_path, 'r')
    frames = audiofile.getnframes()
    rate = audiofile.getframerate()
    duration = frames / float(rate)
    return duration

# IF path is not exist, create the path
def path_is_exists(dirs):
    if not os.path.exists(dirs):
        os.makedirs(dirs)

# A VAD function from didi
def do_vad_didi(args):
    print(args)
    vad_didi_dir = args.vad_didi_dir
    infile_path = args.infile_path
    # the second param is arbitrary
    # ./output/modelvad ./conf/modelvad.conf ${audio_list} ${res} > test.log
    current_dir = os.getcwd()
    os.chdir(vad_didi_dir)
    os.system("sh run.sh %s %s" % (infile_path, "sss"))
    os.chdir(current_dir) 
    #os.system("%s %s %s %s > vad.log" % (os.path.join(vad_didi_dir, "output", "modelvad"), os.path.join(vad_didi_dir, "conf", "modelvad.conf"), infile_path, "sss"))

# VAD operation
def do_vad(args):
    print(args)
    vad_mode = args.vad_mode # "wav_path": input a file with each wav path; "dir_path":dir
    infile_path = args.infile_path
    tr_data_dir = args.tr_data_dir
    audio_str_pattern = args.audio_str_pattern
    output_dir = args.output_dir
    
    path_is_exists(output_dir)

    if vad_mode == "wav_path" and infile_path != "":
        audio_list = list_autio_path_from_file(infile_path)
    elif vad_mode == "dir_path" and tr_data_dir != "" and audio_str_pattern != "":
        audio_list = list_audio_path(tr_data_dir, audio_str_pattern)
    else:
        print("the param 'vad_mode' is error!")
        exit()
	
	#print(audio_list[0])
    for i in range(len(audio_list)):
        audio_name = audio_list[i]
        #if i == 3:
        #    break
        #wav_name = audio_name.split('/')[-1].split('.')[0]
        #output_feature_path = '/home/luban/gemeng/drunk_asr/feature/split_drunk/ch1/' + wav_name + '.arff'
        os.system('python ./vadtools/drunk_vad.py 2 %s %s' % (audio_name.replace('\n',''), output_dir))
        print('python ./vadtools/drunk_vad.py 3 %s %s' % (audio_name.replace('\n',''), output_dir))

def pcm2wav(args):
    print(args)
    pcm_wavs_dir = args.pcm_wavs_dir
    pcm_wav_str_pattern = args.pcm_wav_str_pattern
    output_dir_after_pcm2wav = args.output_dir_after_pcm2wav

    path_is_exists(output_dir_after_pcm2wav)

    pcm_audio_list = list_audio_path_2(pcm_wavs_dir, pcm_wav_str_pattern)
    print(pcm_audio_list)
    for pcm_audio_item in pcm_audio_list:
       wav_file = pcm_audio_item.split("/")[-1]+".wav"
       f = open(pcm_audio_item, "rb")
       str_data = f.read()
       wave_out=wave.open(os.path.join(output_dir_after_pcm2wav, wav_file),'wb')
       wave_out.setnchannels(1)
       wave_out.setsampwidth(2)
       wave_out.setframerate(8000)
       wave_out.writeframes(str_data)


# generate the final opensmile config file (the version of opensmile is 2.3)
def generate_opensmile_conf(args):
    print(args)
    opensmile_dir = args.opensmile_dir
    output_opensmile_bat_conf_name = args.output_opensmile_bat_conf_name
    tr_vad_data_dir = args.tr_vad_data_dir
    audio_str_pattern = args.audio_str_pattern
    opensmile_feature_output_path = args.opensmile_feature_output_path

    path_is_exists(opensmile_feature_output_path)
    # basic config params
    #opensmile_dir = '/home/luban/gemeng/drunk_asr/tools/opensmile-2.3.0'
    runfile_path = opensmile_dir + '/bin/linux_x64_standalone_static/SMILExtract'
    #conf_path = opensmile_dir + '/config/IS09_emotion.conf'
    conf_path = opensmile_dir + '/config/IS09_emotion_only_pitch.conf'
    #conf_path = opensmile_dir + '/config/gemaps/GeMAPSv01a.conf'
    #output_feature_path = '/home/luban/gemeng/drunk_asr/feature/' + audio_name + '.arff'
    output_opensmile_extract_path = opensmile_dir + '/' + output_opensmile_bat_conf_name
    # generate process details
    audio_list = list_audio_path(tr_vad_data_dir, audio_str_pattern)
    print(audio_list[0])
    extracted_idx = 0
    for i in range(len(audio_list)):
        audio_name = audio_list[i]
        #wav_name = audio_name.split('/')[-1].split('.')[0]
        wav_name = os.path.splitext(audio_name.split("/")[-1])[0]
        #output_feature_path = '/home/luban/gemeng/drunk_asr/feature/split_drunk/ch1/' + wav_name + '.arff'
        #output_feature_path = opensmile_feature_output_path + '1.arff'
        output_feature_path = opensmile_feature_output_path + wav_name + '.arff'
        # the length of the current audio
        duration = get_audio_duration(audio_name)
        #if duration < 1.0:
        #    continue
        #print(duration)
        tmp_start = 0
        tmp_skip = 0.050
        segment_length = 0.265
        count = 0
        #if i == 3:
        #    break
        while(1):
            if duration < segment_length:
                str = '%s -C %s -I %s -O %s -start %f -end %f -l 0\n' % (runfile_path, conf_path, audio_name, output_feature_path, tmp_start, duration)
                with open(output_opensmile_extract_path, 'a+') as f:
                    f.write(str)
                break 
            else:
                tmp_end = tmp_start + segment_length
                if tmp_end > duration:
                    break
                count = count + 1
                # generate the commond string
                str = '%s -C %s -I %s -O %s -start %f -end %f -l 0\n' % (runfile_path, conf_path, audio_name, output_feature_path, tmp_start, tmp_end)
                with open(output_opensmile_extract_path, 'a+') as f:
                    f.write(str)
            tmp_start = tmp_start + tmp_skip

        extracted_idx = extracted_idx + 1
        print(extracted_idx)

    f.close()
   #os.system("chmod +x %s" % (output_opensmile_extract_path))

# generate the final opensmile config file (The function only for "pitch" feature, the version of opensmile is 2.3)
def generate_opensmile_conf_pitch(args):
    print(args)
    opensmile_dir = args.opensmile_dir
    output_opensmile_bat_conf_name = args.output_opensmile_bat_conf_name
    tr_vad_data_dir = args.tr_vad_data_dir
    audio_str_pattern = args.audio_str_pattern
    opensmile_feature_output_path = args.opensmile_feature_output_path

    path_is_exists(opensmile_feature_output_path)
    # basic config params
    #opensmile_dir = '/home/luban/gemeng/drunk_asr/tools/opensmile-2.3.0'
    runfile_path = opensmile_dir + '/bin/linux_x64_standalone_static/SMILExtract'
    conf_path = opensmile_dir + '/config/IS09_emotion_only_pitch.conf'
    #conf_path = opensmile_dir + '/config/gemaps/GeMAPSv01a.conf'
    #output_feature_path = '/home/luban/gemeng/drunk_asr/feature/' + audio_name + '.arff'
    output_opensmile_extract_path = opensmile_dir + '/' + output_opensmile_bat_conf_name
    # generate process details
    audio_list = list_audio_path(tr_vad_data_dir, audio_str_pattern)
    print(audio_list[0])
    extracted_idx = 0
    for i in range(len(audio_list)):
        audio_name = audio_list[i]
        #wav_name = audio_name.split('/')[-1].split('.')[0]
        wav_name = os.path.splitext(audio_name.split("/")[-1])[0]
        #output_feature_path = '/home/luban/gemeng/drunk_asr/feature/split_drunk/ch1/' + wav_name + '.arff'
        #output_feature_path = opensmile_feature_output_path + '1.arff'
        output_feature_path = opensmile_feature_output_path + wav_name + '.csv'
        # the length of the current audio
        duration = get_audio_duration(audio_name)
        #if duration < 1.0:
        #    continue
        #print(duration)
        tmp_start = 0
        tmp_skip = 0.050
        segment_length = 0.265
        count = 0
        #if i == 3:
        #    break
        str = '%s -C %s -I %s -csvoutput %s -l 0\n' % (runfile_path, conf_path, audio_name, output_feature_path)
        with open(output_opensmile_extract_path, 'a+') as f:
            f.write(str)
        #while(1):
        #    if duration < segment_length:
        #        str = '%s -C %s -I %s -O %s -start %f -end %f -l 0\n' % (runfile_path, conf_path, audio_name, output_feature_path, tmp_start, duration)
        #        with open(output_opensmile_extract_path, 'a+') as f:
        #            f.write(str)
        #        break 
        #    else:
        #        tmp_end = tmp_start + segment_length
        #        if tmp_end > duration:
        #            break
        #        count = count + 1
        #        # generate the commond string
        #        str = '%s -C %s -I %s -O %s -start %f -end %f -l 0\n' % (runfile_path, conf_path, audio_name, output_feature_path, tmp_start, tmp_end)
        #        with open(output_opensmile_extract_path, 'a+') as f:
        #            f.write(str)
        #    tmp_start = tmp_start + tmp_skip

        extracted_idx = extracted_idx + 1
        print(extracted_idx)

    f.close()
   #os.system("chmod +x %s" % (output_opensmile_extract_path))

# generate opensmile config file from entire wav file 
def generate_opensmile_conf_from_wav(args):
    print(args)
    opensmile_dir = args.opensmile_dir
    output_opensmile_bat_conf_name = args.output_opensmile_bat_conf_name
    tr_vad_data_dir = args.tr_vad_data_dir
    audio_str_pattern = args.audio_str_pattern
    opensmile_feature_output_path = args.opensmile_feature_output_path
    #print("tttt")
    path_is_exists(opensmile_feature_output_path)
    #print("eeee")
    # basic config params
    #opensmile_dir = '/home/luban/gemeng/drunk_asr/tools/opensmile-2.3.0'
    runfile_path = opensmile_dir + '/bin/linux_x64_standalone_static/SMILExtract'
    conf_path = opensmile_dir + '/config/IS09_emotion.conf'
    #output_feature_path = '/home/luban/gemeng/drunk_asr/feature/' + audio_name + '.arff'
    output_opensmile_extract_path = opensmile_dir + '/' + output_opensmile_bat_conf_name
    #path_is_exists(output_opensmile_extract_path)
    # generate process details
    audio_list = list_audio_path(tr_vad_data_dir, audio_str_pattern)
    #print(audio_list[0])
    extracted_idx = 0
    for i in range(len(audio_list)):
        audio_name = audio_list[i]
        #print(audio_name)
        wav_names = audio_name.split('/')[-1].split('.')
        #print(wav_names)
        #wav_name = wav_names[0] + "_" + wav_names[2]
        wav_name = wav_names[0]

        #output_feature_path = '/home/luban/gemeng/drunk_asr/feature/split_drunk/ch1/' + wav_name + '.arff'
        output_feature_path = opensmile_feature_output_path + 'all_feats.arff'
        # the length of the current audio
        duration = get_audio_duration(audio_name)
        #print(duration)
        # if duration < 1.0:
        #     continue
        tmp_start = 0
        # # tmp_skip = 0.05
        # # segment_length = 0.265
        # count = 0
        # #if i == 3:
        # #    break
        # # generate the commond string
        # if duration < segment_length:
        #     str = '%s -C %s -I %s -O %s -start %f -end %f -l 0\n' % (runfile_path, conf_path, audio_name, output_feature_path, tmp_start, duration)
        #     with open(output_opensmile_extract_path, 'a+') as f:
        #         f.write(str)
        #     break 
        # else:
        #print("ffff")
        str = '%s -C %s -I %s -O %s -start %f -end %f -l 0\n' % (runfile_path, conf_path, audio_name, output_feature_path, tmp_start, duration)
        with open(output_opensmile_extract_path, 'a+') as f:
            f.write(str)
        f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')

    parser_vad = subparsers.add_parser('vad')
    parser_vad.add_argument('--vad_mode', type=str, required=True)
    parser_vad.add_argument('--infile_path', type=str, required=False)
    parser_vad.add_argument('--tr_data_dir', type=str, required=False)
    parser_vad.add_argument('--audio_str_pattern', type=str, required=False)
    parser_vad.add_argument('--output_dir', type=str, required=True)
   
    parser_vad = subparsers.add_parser('vad_didi')
    parser_vad.add_argument('--vad_didi_dir', type=str, required=True)
    parser_vad.add_argument('--infile_path', type=str, required=False)
    
    parser_vad = subparsers.add_parser('pcm2wav')
    parser_vad.add_argument('--pcm_wavs_dir', type=str, required=True)
    parser_vad.add_argument('--pcm_wav_str_pattern', type=str, required=False)
    parser_vad.add_argument('--output_dir_after_pcm2wav', type=str, required=False)
    
    parser_opensmile = subparsers.add_parser('create_opensmile_bat_conf')
    parser_opensmile.add_argument('--opensmile_dir', type=str, required=True)
    parser_opensmile.add_argument('--output_opensmile_bat_conf_name', type=str, required=True)
    parser_opensmile.add_argument('--tr_vad_data_dir', type=str, required=True)
    parser_opensmile.add_argument('--audio_str_pattern', type=str, required=True)
    parser_opensmile.add_argument('--opensmile_feature_output_path', type=str, required=True)


    args = parser.parse_args()

    if args.mode == 'vad':
        do_vad(args)
    elif args.mode == 'vad_didi':
        do_vad_didi(args)
    elif args.mode == 'pcm2wav':
        pcm2wav(args)
    elif args.mode == 'create_opensmile_bat_conf':
        #generate_opensmile_conf(args)
        #generate_opensmile_conf_from_wav(args)
        generate_opensmile_conf_pitch(args)
    else:
        raise Exception("Error!!")
    # configure
    #tr_data_dir = '/nfs/project/yangguilin/drunk/wav_channel_2/'
    #audio_str_pattern = '.wav'
    #do_vad(tr_data_dir, audio_str_pattern)

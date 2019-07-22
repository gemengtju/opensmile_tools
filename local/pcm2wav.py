#/usr/bin/python

import wave
import os

# get the duration of an audio
def get_audio_duration(audio_path):
    audiofile = wave.open(audio_path, 'r')
    frames = audiofile.getnframes()
    rate = audiofile.getframerate()
    duration = frames / float(rate)
    return duration

# list the paths of all audios
def list_audio_path(file_dir, str_pattern):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == str_pattern:
                L.append(os.path.join(root, file))
    return L

if __name__ == "__main__":
    audio_list = list_audio_path("./test",".pcm")
    for audio_item in audio_list:
       #print(audio_item.split("/")[-1]+".wav")
       wav_file = audio_item.split("/")[-1]+".wav"
       f = open(audio_item, "rb")
       str_data = f.read()
       wave_out=wave.open(wav_file,'wb')
       wave_out.setnchannels(1)
       wave_out.setsampwidth(2)
       wave_out.setframerate(8000)
       wave_out.writeframes(str_data)
        #print(a)
        #if a < 1 :
        #    print(a)
            #print("delete")
        #    os.system("rm -r %s" % (audio_item))

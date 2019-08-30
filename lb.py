import io
import os
import subprocess
import pandas as pd
import fnmatch

main_dir = "LibriSpeech_dataset/"


# getting all file paths end with some thing like '.flac', '.wav' , 'trains.txt'
def all_file_paths(data_path, extencetion):
    all_paths = [os.path.join(dirpath, f)
                    for dirpath, dirnames, files in os.walk(data_path)
                    for f in fnmatch.filter(files, '*'+extencetion)]
    return all_paths
# coverting '.flac' to '.wav' with sample_rate = 16000
def converting_flac_to_wav(flac_path, sample_rate, remove=False):
    # copy the '.flac' path and rename as '.wav'
    wav_path = flac_path.replace(".flac", ".wav")
    # convert .flac to .wav
    subprocess.call(["sox {}  -r {} -b 16 -c 1 {}".format(flac_path, str(sample_rate),wav_path)], shell=True)
    # delete .flac file
    if remove == True:
        os.remove(flac_path)
    return wav_path

# find out index of uniq audio
def index_of_uniq_audio(wav_path):
    index = wav_path.split('/')[-1].split('.')[0]
    return index


# duration of each audio file
def duration_of_audio(path):
    duration_file_paths = float(subprocess.check_output(
        ['soxi -D \"%s\"' % path.strip()], shell=True))
    return duration_file_paths

# for each audio file we add text file using uniqe index values
def append_text_to_audio(text_paths, df):
    for path in text_paths:
        transcriptions = open(path).read().strip().split("\n")
        for sent in transcriptions:
            ind = sent.split(' ',1)
            if ind[0] in df.index:
                df.loc[ind[0], 'text'] = ind[1]
    return df

# create dataframe with index and audio paths
def create_df(df_name , data_path , sample_rate, done=True):
    # all flac file paths
    flac_paths = all_file_paths(data_path, '.flac')
    # all text file paths
    text_paths = all_file_paths(data_path, 'trans.txt')
    # create lists
    if done == True:
        wav_paths = all_file_paths(data_path, '.wav')
    else:
        wav_paths = []
        for path in flac_paths:
            wav_path = converting_flac_to_wav(path, sample_rate=sample_rate, remove=False)
            wav_path.append(wav_path)
    indexes = [index_of_uniq_audio(i) for i in wav_paths]
    durations = [duration_of_audio(i) for i in wav_paths]

    # create dataframe
    df = {'wav_paths': wav_paths, 'durations':durations}

    df = pd.DataFrame(df , index = indexes)

    # adding new column text corresponding audio

    df = append_text_to_audio(text_paths, df)

    df.to_csv(df_name, sep=',', index_label='indexes')

    return df


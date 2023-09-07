import librosa
import glob
import os
import matplotlib.pyplot as plt 
import csv
import numpy as np


data_folder = r'C:\Users\yair\stt_trauma\data'
# for audio_path in glob.glob(os.path.join(data_folder,'*.3gp')):
#     x , sr = librosa.load(audio_path)
#     mfccs = librosa.feature.mfcc(y=x, sr=sr)
#     print(mfccs.shape)
#     #Displaying  the MFCCs:
#     librosa.display.specshow(mfccs, sr=sr, x_axis='time')
#     plt.savefig(audio_path.replace('.3gp','_mfcc.png'))
# plt.close()


header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
for i in range(1, 21):
    header += f' mfcc{i}'
header += ' label'
header = header.split()

with open(os.path.join(data_folder,'data.csv'), 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)
    types_list = 'tach bla'.split()
    for g in types_list:
        for filename in glob.glob(os.path.join(data_folder,g,'*.3gp')):
            y, sr = librosa.load(filename, mono=True, duration=30)
            chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
            rmse = librosa.feature.rms(y=y)[0]
            spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
            spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            zcr = librosa.feature.zero_crossing_rate(y)
            mfcc = librosa.feature.mfcc(y=y, sr=sr)
            to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'    
            for e in mfcc:
                to_append += f' {np.mean(e)}'
            to_append += f' {g}'
            writer = csv.writer(file)
            writer.writerow(to_append.split())


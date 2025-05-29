'''
Copied from espnet: https://github.com/Wendison/VQMIVC/preprocess.py
'''

from pwg_vqmivc_spectrogram import logmelspectrogram
import numpy as np
from joblib import Parallel, delayed
import librosa
import soundfile as sf
import os
from glob import glob
from tqdm import tqdm
import random
import json
import resampy
import pyworld as pw

def extract_logmel(wav_path, sr=16000):
    # wav, fs = librosa.load(wav_path, sr=sr)
    wav, fs = sf.read(wav_path)
    wav, _ = librosa.effects.trim(wav, top_db=60)
    if fs != sr:
        wav = resampy.resample(wav, fs, sr, axis=0)
        fs = sr
    # duration = len(wav)/fs
    assert fs == 16000
    peak = np.abs(wav).max()
    if peak > 1.0:
        wav /= peak
    mel = logmelspectrogram(
                x=wav,
                fs=fs,
                n_mels=80,
                n_fft=400,
                n_shift=160,
                win_length=400, 
                window='hann',
                fmin=80,
                fmax=7600,
            )
    
    tlen = mel.shape[0]
    frame_period = 160/fs*1000
    f0, timeaxis = pw.dio(wav.astype('float64'), fs, frame_period=frame_period)
    f0 = pw.stonemask(wav.astype('float64'), f0, timeaxis, fs)
    f0 = f0[:tlen].reshape(-1).astype('float32')
    nonzeros_indices = np.nonzero(f0)
    lf0 = f0.copy()
    lf0[nonzeros_indices] = np.log(f0[nonzeros_indices]) # for f0(Hz), lf0 > 0 when f0 != 0
                
    wav_name = wav_path.split('/')[-2] + '_' + os.path.basename(wav_path)[:-4]
    # print(wav_name, mel.shape, duration)
    return wav_name, mel, lf0, mel.shape[0]


def normalize_logmel(wav_name, mel, mean, std):
    mel = (mel - mean) / (std + 1e-8)
    return wav_name, mel


def save_one_file(save_path, arr):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, arr)


def save_logmel(save_root, wav_name, melinfo, mode):
    mel, lf0, mel_len = melinfo
    spk = wav_name.split('_')[0]
    mel_save_path = f'{save_root}/{mode}/mels_200/{spk}/{wav_name}.npy'
    lf0_save_path = f'{save_root}/{mode}/lf0_200/{spk}/{wav_name}.npy'
    save_one_file(mel_save_path, mel)
    save_one_file(lf0_save_path, lf0)
    return mel_len, mel_save_path, lf0_save_path


tsv_base_dir = '/data0/yfliu/lrs3/top200_data/'
save_root = '/data0/yfliu/lrs3/pwg_vqmivc'
os.makedirs(save_root, exist_ok=True)

def read_names(tsv_path):
    paths = []
    names = []
    with open(tsv_path, 'r') as fr:
        fr.readline()
        for line in fr.readlines():
            path = line.strip().split('\t')[2]
            paths.append(path)
            name = path.split('/')[-2] + '_' + os.path.basename(path)[:-4]
            names.append(name)
    return paths, names

train_all_wavs, train_wavs_names = read_names(tsv_base_dir+'train.tsv')
wavs_names = train_wavs_names  # test should not be included here.
print(len(wavs_names))

# extract log-mel
print('extract log-mel...')

all_wavs = []
for i in train_all_wavs:
    all_wavs.append(i)


results = Parallel(n_jobs=-1)(delayed(extract_logmel)(wav_path) for wav_path in tqdm(all_wavs))
wn2mel = {}
for r in results:
    wav_name, mel, lf0, mel_len = r
    # print(wav_name, mel.shape, duration)
    wn2mel[wav_name] = [mel, lf0, mel_len]

# normalize log-mel
print('normalize log-mel...')
mels = []
spk2lf0 = {}
for wav_name in train_wavs_names:
    mel, _, _ = wn2mel[wav_name]
    mels.append(mel)

mels = np.concatenate(mels, 0)
mean = np.mean(mels, 0)
std = np.std(mels, 0)
mel_stats = np.concatenate([mean.reshape(1,-1), std.reshape(1,-1)], 0)
np.save(f'{save_root}/mel_stats_200.npy', mel_stats)
    



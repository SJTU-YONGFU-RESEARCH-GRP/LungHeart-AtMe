#!/usr/bin/python3
# from tqwt_tools.tqwt_tools import DualQDecomposition
# from utils import read_split_data, write_pickle
import shutil
import random
from pydub import AudioSegment
AudioSegment.converter = "/home/kaswary/ffmpeg-git-20220422-amd64-static/ffmpeg"
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import librosa
import scipy.io.wavfile as wav
from scipy.signal import butter, lfilter, firwin
from scipy import stats
import librosa.display as libdisplay
import multiprocessing as mp
import time
import datetime
import csv
import sys
from PIL import Image
# Image.LOAD_TRUNCATED_IMAGES = True
import torch
from torch.utils.data import Dataset
from Args import args
from utils import *
import pywt
import soundfile
import subprocess

            
def moveFileOfficial(file_dir, save_dir, official_txt):
    """
    seprate trainset and testset. official
    Args:
        file_dir: source direction
        save_dir: target direction
        official_txt: official train-test speration txt file
    """
    images = []
    save_train, save_test = [os.path.join(save_dir, 'train'), os.path.join(save_dir, 'test')]
    save_paths = [os.path.join(save_train, 'WAV'), os.path.join(save_test, 'WAV'),\
                  os.path.join(save_train, 'TXT'), os.path.join(save_test, 'TXT')]
    save_paths.extend([save_train, save_test])
    for p in save_paths:
        ensure_dir_exists(p)
    if os.path.isfile(official_txt):
        with open(official_txt, 'r') as f:
            for line in f.readlines():
                data = line.split('/t/n')
                for str in data:
                    sub_str = str.split('\t')
                if sub_str:
                    images.append(sub_str)
    for image in images:
        print(image)
        save_root = save_train if 'train' in image[1] else save_test
        initial_txt, initial_wav = [os.path.join(file_dir, 'TXT', image[0] + '.txt'), os.path.join(file_dir, 'WAV', image[0] + '.wav')]
        target_txt, target_wav = [os.path.join(save_root,'TXT', image[0] + '.txt'), os.path.join(save_root,'WAV', image[0] + '.wav')]
        if not os.path.exists(initial_txt) or not os.path.exists(initial_wav):
            print("{} not exist".format(initial_wav))
            continue
        shutil.copy(initial_txt, target_txt)
        shutil.copy(initial_wav, target_wav)
        print("Move {} to {}".format(initial_wav, target_wav))
   
   
def clip_cycle(i_dir, n_dir):
    """
    clip the record into breath cycle
    dir : trainset/testset record path
    new_dir: breath cycle save path
    """
    wav_dir = os.path.join(i_dir, 'WAV')
    txt_dir = os.path.join(i_dir, 'TXT')
    for file in os.listdir(txt_dir):
        txt_path = os.path.join(txt_dir, file)
        wav_path = os.path.join(wav_dir, file[:-4]+'.wav')
        if os.path.exists(wav_path):
            time = np.loadtxt(txt_path)[:, 0:2]
            sound = AudioSegment.from_wav(wav_path)
            for i in range(time.shape[0]):
                target_wav_path = os.path.join(n_dir, file[:-4] + str(i) + '.wav')
                start_time = time[i, 0] * 1000
                stop_time = time[i, 1] * 1000
                cycle = sound[start_time:stop_time]
                cycle.export(target_wav_path, format="wav")
                print("Generate cycle: {}".format(target_wav_path))
        else:
            print("{} not exist".format(wav_path))
            continue


def moveFileClasses(file_dir, save_dir):
    classes = {
        "Normal": 0, 
        "Crackle": 1,
        "Wheeze": 2,
        "CplueW":3,
        "HeartNormal":4,
        "HeartAbnormal":5
        }
    for tt in ["train", "test"]:
        save_tt_dir = os.path.join(save_dir, tt)
        ensure_dir_exists(save_tt_dir)
        for v in classes:
            ensure_dir_exists(os.path.join(save_tt_dir, v))
        if "ICBHI" in file_dir:
            root_dir = os.path.join(file_dir, tt)
            labels_dir = os.path.join(root_dir, "TXT")
            cycles_dir = os.path.join(root_dir, "Cycles")
            class_type = 0
            for label_file in os.listdir(labels_dir):
                label_file_path = os.path.join(labels_dir, label_file)
                with open(label_file_path, 'r') as f:
                    for idx, line in enumerate(f.readlines()):
                        data = line.strip().split('\t')
                        if data[2] == "0" and data[3] == "0":
                            class_type = 0
                        elif data[2] == "1" and data[3] == "0":
                            class_type = 1
                        elif data[2] == "0" and data[3] == "1":
                            class_type = 2
                        elif data[2] == "1" and data[3] == "1":
                            class_type = 3
                        else:
                            print("Error")
                            exit(0)
                        cycle_file_path = os.path.join(cycles_dir, label_file[:-4]+str(idx)+".wav")
                        target_file_path = os.path.join(save_tt_dir, str(class_type), label_file[:-4]+str(idx)+".wav")
                        if not os.path.exists(target_file_path):
                            shutil.copy(cycle_file_path, target_file_path)
                            print("Move", cycle_file_path, "to", target_file_path)
        else:
            labels_csv = os.path.join(file_dir, "label_{}.csv".format(tt)) if tt == "train" else os.path.join(file_dir, "label_{}.csv".format("valid"))
            wavs_dir = os.path.join(file_dir, "WAV", "wav_{}".format(tt)) if tt == "train" else os.path.join(file_dir, "WAV", "wav_{}".format("valid"))
            class_type = 0
            with open(labels_csv, 'r') as f:
                for idx, line in enumerate(f.readlines()):
                    data = line.strip().split(',')
                    if data[1] == '1':
                        class_type = 4
                    else:
                        class_type = 5
                    cycle_file_path = os.path.join(wavs_dir, data[0]+".wav")
                    target_file_path = os.path.join(save_tt_dir, str(class_type), data[0]+".wav")
                    if not os.path.exists(target_file_path):
                        shutil.copy(cycle_file_path, target_file_path)
                        print("Move", cycle_file_path, "to", target_file_path)
                    
    
def read_dir(origin_dir, target_dir):
    wav_paths = []
    mk_paths = []
    target_file_paths = []
    file_dir = os.listdir(origin_dir)
    for tt in file_dir:
        tt_path = os.path.join(origin_dir, tt)
        if not os.path.isdir(tt_path):
            continue
        target_tt_path = os.path.join(target_dir, tt)
        mk_paths.append(target_tt_path)
        for label in os.listdir(tt_path):
            lb_path = os.path.join(tt_path, label)
            target_lb_path = os.path.join(target_tt_path, label)
            mk_paths.append(target_lb_path)
            for file in os.listdir(lb_path):
                file_path = os.path.join(lb_path, file)
                wav_paths.append(file_path)
                target_file_path = os.path.join(target_lb_path, file[:-3] + "png")
                target_file_paths.append(target_file_path)
    return wav_paths, mk_paths, target_file_paths


def ensure_dir_exists(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        

def sc_aug(path1, path2, target_p="", alpha=args.alpha, save=0):
    target_p_aug = target_p[:-4] + "_sc" + ".png"
    if os.path.exists(target_p_aug):
        # print("            Same Class Augmented Wave File exists")
        return
    lam = np.random.beta(alpha, alpha)
    sig1, fs1 = librosa.load(path1)
    sig2, fs2 = librosa.load(path2)
    len1 = len(sig1)
    len2 = len(sig2)
    len_new = min([len1, len2])
    sig_new = lam * sig1[:len_new] + (1 - lam) * sig2[:len_new]
    fs_new = fs1
    if save:
        soundfile.write(target_p, sig_new, fs_new)
    return [[sig_new, fs_new, target_p_aug]]


def ts_aug(path, ratio, target_p="", save=0):
    if os.path.exists(target_p):
        # print("            Time Shift Augmented Wave File exists")
        return
    sig, fs = librosa.load(path)
    length = len(sig)
    # shutil.copy(path, target_p)
    new = []
    if ratio == 0:
        return
    for i in range(ratio):
        target_p_aug = target_p[:-4] + "_ts" + str(i+1) + ".png"
        nb_shifts = np.random.randint(0, length)
        sig_new = np.roll(sig, nb_shifts)
        new.append([sig_new, fs, target_p_aug])
        if save:
            soundfile.write(target_p_aug, sig_new, fs)
    return new


def denoise_sig(signals, threshold):
    w = pywt.Wavelet('db8')   # 选用Daubechies8小波
    sig_new = []
    for [sig, fs, target_p] in signals:
        maxlev = pywt.dwt_max_level(len(sig), w.dec_len)
        print("maximum level is " + str(maxlev))
        # Decompose into wavelet components, to the level selected:
        coeffs = pywt.wavedec(sig, 'db8', level=maxlev)  # 将信号进行小波分解
        for i in range(1, len(coeffs)):
            coeffs[i] = pywt.threshold(coeffs[i], threshold * max(coeffs[i]))  # 将噪声滤波
        datarec = pywt.waverec(coeffs, 'db8')
        sig_new.append([datarec, fs, target_p])
    return sig_new


def stft_sig(signal, inter_p):
    print("----------- STFT ----------")
    sig = signal[0]
    fs = signal[1]
    target_p = signal[2]
    name = target_p.split('/')[-1]
    # stft_path = os.path.join(inter_p, "stft.png")
    stft_path = target_p
    stft = librosa.stft(sig, n_fft=int(0.02 * 4000), hop_length=int(0.01 * 4000), window='hann')
    if fs > 4000:
        libdisplay.specshow(librosa.amplitude_to_db(stft[0:int(len(stft) / 2), :], ref=np.max), y_axis='log',
                            x_axis='time')
    else:
        libdisplay.specshow(librosa.amplitude_to_db(stft, ref=np.max), y_axis='log', x_axis='time')
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.savefig(stft_path)
    print("-----------save stft: ", name, "----------")
    plt.clf()
    return stft_path


def mfcc_sig(signal, inter_p):
    print("----------- MFCC ----------")
    sig = signal[0]
    fs = signal[1]
    target_p = signal[2]
    name = target_p.split('/')[-1]
    mfcc_path = os.path.join(inter_p, "mfcc.png")
    mfcc = librosa.feature.mfcc(y=sig, sr=fs, n_mfcc=128)
    if fs > 4000:
        libdisplay.specshow(librosa.amplitude_to_db(mfcc[0:int(len(mfcc) / 2), :], ref=np.max), x_axis='time',
                            y_axis='mel')
    else:
        libdisplay.specshow(librosa.amplitude_to_db(mfcc, ref=np.max), x_axis='time', y_axis='mel')
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.savefig(mfcc_path)
    print("-----------save mfcc: ", name, "----------")
    plt.clf()
    return mfcc_path


def merge(spec_path, stft_path, mfcc_path):
    print("----------- Merge ----------")
    name = spec_path.split('/')[-1]
    stft_fig = np.array(Image.open(stft_path).convert('L'))
    mfcc_fig = np.array(Image.open(mfcc_path).convert('L'))
    shape = stft_fig.shape
    mg = Image.fromarray(np.zeros(shape)).convert('L')
    # mg = Image.fromarray(stft_fig + mfcc_fig)
    stft_channel = Image.fromarray(stft_fig)
    mfcc_channel = Image.fromarray(mfcc_fig)
    merge_fig = Image.merge("RGB", (stft_channel, mfcc_channel, mg))
    merge_fig.save(spec_path)
    print("-----------save merge spectrogram: ", name, "----------")
    plt.clf()
    

def wd_stft_mfcc_merge(root, target_dir, inter_p, threshold=args.denoise_threshold, aug=0):
    wavs_paths, mk_paths, target_file_paths = read_dir(root, target_dir)
    for ps in mk_paths:
        print("Make directory: ", ps)
        ensure_dir_exists(ps)
    ts_new = []
    sc_new = []

    for idx, wav in enumerate(wavs_paths):
        print("-----------process num.: ", idx, "----------")
        name = wav.split('/')[-1]
        spec_path = target_file_paths[idx]
        if not os.path.exists(spec_path):
            sig, fs = librosa.load(wav)
            signals = [[sig, fs, spec_path]]
            # data augmentation
            if aug:
                sc_ratio_list = args.sc_aug_ratio
                ts_ratio_list = args.ts_aug_ratio
                print("----------- Time Shift and Same Class-Augment num.", str(idx), " Wave File: ", wav, " -----------")
                if wav.split('/')[-3] == "train":
                    label = wav.split('/')[-2]
                    if label == "cplusw":
                        ts_ratio = ts_ratio_list[0]
                        sc_ratio = sc_ratio_list[0]
                    elif label == "crackle":
                        ts_ratio = ts_ratio_list[1]
                        sc_ratio = sc_ratio_list[1]
                    elif label == "habnormal":
                        ts_ratio = ts_ratio_list[2]
                        sc_ratio = sc_ratio_list[2]
                    elif label == "hnormal":
                        ts_ratio = ts_ratio_list[3]
                        sc_ratio = sc_ratio_list[3]
                    elif label == "lnormal":
                        ts_ratio = ts_ratio_list[4]
                        sc_ratio = sc_ratio_list[4]
                    else:
                        ts_ratio = ts_ratio_list[5]
                        sc_ratio = sc_ratio_list[5]
                else:
                    ts_ratio = 0
                    sc_ratio = 0
                ts_new = ts_aug(wav, ts_ratio, spec_path)
                if sc_ratio:
                    sc_idx = np.random.randint(0, len(wavs_paths))
                    wav_sc = wavs_paths[sc_idx]
                    sc_new = sc_aug(wav, wav_sc, spec_path)
                if ts_new:
                    if sc_new:
                        signals = np.concatenate([signals, ts_new, sc_new])
                    else:
                        signals = np.concatenate([signals, ts_new])
                print("----------- Augmented signals obtained -----------")

            # denoise
            print("----------- Denoise num.", str(idx), " Wave File: ", wav, " -----------")
            signals = denoise_sig(signals, threshold)
            # if sig.size % 2 != 0:
            #     sig = sig[:-1]

            for sig in signals:
                stft_path = stft_sig(sig, inter_p)
                mfcc_path = mfcc_sig(sig, inter_p)
                merge(sig[2], stft_path, mfcc_path)

    print("----------- Completed!! -----------")


def statistcs(root):
    train = os.path.join(root, "train")
    test = os.path.join(root, "test")
    histogram = os.path.join(root, "data_distribution_histogram.png")
    classes = os.path.join(root, "classes.json")
    lists = os.path.join(root, "train_val_lists.p")
    _ = read_split_data(train, test, histogram, classes, lists)


if __name__ == "__main__":
    # # seprate trainset and testset ----- ICBHI
    # file_path = 'Dataset/ICBHI2017/train-test.txt'
    # initial_dir = 'Dataset/ICBHI2017'
    # target_dir = initial_dir
    # ensure_dir_exists(target_dir)
    # moveFileOfficial(initial_dir, target_dir, file_path)
    
    # # clip the record into breath cycles ----- ICBHI
    # root_dirs = {
    #     "train": 'Dataset/ICBHI2017/train',
    #     "test": 'Dataset/ICBHI2017/test'
    # }
    # cycles_dirs = {
    #     "train": 'Dataset/ICBHI2017/train/Cycles',
    #     "test": 'Dataset/ICBHI2017/test/Cycles'
    # }
    # for r_dir, c_dir in zip(root_dirs, cycles_dirs):
    #     ensure_dir_exists(cycles_dirs[c_dir])
    #     clip_cycle(root_dirs[r_dir], cycles_dirs[c_dir])

    # move wav files to corresponding folders
    dataset_dirs = {
        "lung": "Dataset/ICBHI2017",
        "heart": "Dataset/PhysioNet2016"
    }
    classes_dir = "Dataset/TT_Classes"
    ensure_dir_exists(classes_dir)
    for i in dataset_dirs:
        moveFileClasses(dataset_dirs[i], classes_dir)
    
    # generate spectrograms
    tmp = "tmp"
    ensure_dir_exists(tmp)
    fig_dir_aug = {
        "merge_dir": "Dataset/aug_denoise_stft_mfcc"
    }
    wd_stft_mfcc_merge(classes_dir, fig_dir_aug["merge_dir"], tmp, aug=1)
    shutil.rmtree(tmp)
    statistcs(fig_dir_aug["merge_dir"])
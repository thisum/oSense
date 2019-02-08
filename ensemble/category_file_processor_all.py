import numpy as np
import pandas as pd
from scipy import pi
import matplotlib.pyplot as plt
from scipy import fftpack
from scipy import hamming
from scipy import signal
import os
from hmmlearn import hmm
import scipy.fftpack
import scipy.stats as st
import shutil

data_points = 2000
T = 1.0 / 35.0

# lbl_list = ['knife_cutting', 'hammer_hammering', 'pen_writing', 'mug_drinking', 'spoon_stirring',
#             'saw_sawing', 'screwdriver_screwing(no-release)', 'screwdriver_screwing(release)', 'knife_chopping', 'bottle_drinking']

def appened_to_file(flie_name, line, trim):
    with open(flie_name, "a") as write_file:
        line = line.replace("[", "").replace("]", "").replace('\'', '')
        if trim:
            line = line.replace(" ", "")
        write_file.write(line + "\n")


def calculate_rms_features(s_d):
    mean = np.mean(s_d)
    std = np.std(s_d)
    min = np.min(s_d)
    max = np.max(s_d)
    medi = np.median(s_d)
    skew = st.skew(s_d)
    kurt = st.kurtosis(s_d)
    rms = np.sqrt(np.mean(np.power(s_d, 2)))

    return (mean, std, min, max, medi, skew, kurt, rms)

def calculate_fft_features(f_d):
    flat = f_d.real.flatten()
    flat.sort()

    n = list(f_d.real).index(flat[-1])
    d_f1 = n*(1.0/(T))/f_d.size
    p_f1 = abs(f_d[n]) ** 2

    n = list(f_d.real).index(flat[-2])
    d_f2 = n*(1.0/(T))/f_d.size
    p_f2 = abs(f_d[n]) ** 2

    n = list(f_d.real).index(flat[-3])
    d_f3 = n*(1.0/(T))/f_d.size
    p_f3 = abs(f_d[n]) ** 2

    n = list(f_d.real).index(flat[-4])
    d_f4 = n*(1.0/(T))/f_d.size
    p_f4 = abs(f_d[n]) ** 2

    n = list(f_d.real).index(flat[-5])
    d_f5 = n*(1.0/(T))/f_d.size
    p_f5 = abs(f_d[n]) ** 2

    if(d_f1 == 0.0):
        d_f1 = d_f2
        d_f2 = d_f3
        d_f3 = d_f4
        d_f4 = d_f5

        p_f1 = p_f2
        p_f2 = p_f3
        p_f3 = p_f4
        p_f4 = p_f5

        n = list(f_d.real).index(flat[-6])
        d_f5 = n * (1.0 / (T)) / f_d.size
        p_f5 = abs(f_d[n]) ** 2


    p_t = np.sum([abs(f_d[x])**2 for x in range(int(np.floor(0.3*T*f_d.size)), int(np.ceil(10*T*f_d.size)))])

    return(d_f1, d_f2, d_f3, d_f4, d_f5, p_f1, p_f2, p_f3, p_f4, p_f5, p_t, p_f1/p_t, p_f2/p_t, p_f3/p_t, p_f4/p_t, p_f5/p_t)


def calculate_features(s_d, f_d, prev_freq, lbl, target_file):
    feature_row = []
    rms_features = calculate_rms_features(s_d)
    fft_features = calculate_fft_features(f_d)

    feature_row.append([x for x in (rms_features + fft_features)])
    feature_row.append( (fft_features[0] / prev_freq))
    feature_row.append(lbl)

    appened_to_file("../data/activity/accel_features/" + target_file + ".txt", str(feature_row), True)

    return fft_features[0]


def process_file(file_path, target_file, lbl_list, W):

    accel_data = pd.read_csv(file_path, header=None)
    accel_data = accel_data.loc[:,:2]
    accel_data = accel_data.sum(axis=1)
    accel_data = accel_data ** (0.5)

    avg = np.mean(accel_data)
    print("avg: " + str(avg))

    for lbl in lbl_list:
        raw_data = pd.read_csv(file_path, header=None)
        raw_data = raw_data[raw_data[3].isin([lbl])]
        raw_data = raw_data.reset_index(drop=True)

        print(lbl + " " + str(raw_data.shape))

        data_points = np.floor(raw_data.shape[0] / 3)

        for j in range(3):
            accel_data = raw_data.loc[j * data_points: (j + 1) * data_points - 1, :2] ** 2
            accel_data = accel_data.sum(axis=1)
            accel_data = accel_data ** (0.5)

            y = [i - avg for i in accel_data]
            prev_freq = 1.0

            for i in range(len(y)):

                # begin = int(i * (W/10))
                begin = i
                if begin + W > len(y):
                    break
                series = y[begin: begin + W]
                hamm = np.hamming(len(series))
                yf = fftpack.fft(series * hamm)

                prev_freq = calculate_features(series, yf, prev_freq, lbl, target_file)


def prepare_test_train_files(lbl_list, W):

    filelist = [f for f in os.listdir("../data/activity/accel_features/")]
    for f in filelist:
        os.remove(os.path.join("../data/activity/accel_features/", f))

    for root, dirs, files in os.walk(
            "/Users/thisum/Documents/Personel_Docs/NUS_MSc/Research/MagicHand_Data/sequence_data/experiment_data/processed/accel_character/raw/"):
        for i, filename in enumerate(files):

            if filename.startswith("."):
                continue

            if 'thisum' in filename:
                print(filename)
                process_file(root + "/" + filename, filename.split(".")[0], lbl_list, W)
                print("\n----------------\n\n")

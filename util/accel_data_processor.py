import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy import fftpack
import os


lbl_list = ['knife_cutting', 'hammer_hammering', 'pen_writing', 'mug_drinking', 'spoon_stirring',
            'saw_sawing', 'screwdriver_screwing(release)', 'knife_chopping', 'bottle_drinking']

def appened_to_file(flie_name, line):
    with open(flie_name, "a") as write_file:
        write_file.write(line)



# make one file with all data
# for root, dirs, files in os.walk("/Users/thisum/Documents/Personel_Docs/NUS_MSc/Research/MagicHand_Data/sequence_data/experiment_data/processed/accel_character/"):
#     for i, filename in enumerate(files):
#
#         if filename.startswith("."):
#             continue
#
#         file = open(root + "/" + filename, "r")
#         lines = file.readlines()
#         for line in lines:
#             appened_to_file("all.csv", line)


filelist = [ f for f in os.listdir("../data/txt_files/per_user_train_test/")]
for f in filelist:
    os.remove(os.path.join("../data/txt_files/per_user_train_test/", f))

# make train and test files for each user for accelerometer data
for root, dirs, files in os.walk("../data/accel_features/"):
    for i, filename in enumerate(files):

        if filename.startswith("."):
            continue

        train = pd.DataFrame()
        test = pd.DataFrame()

        new_name = filename.split(".")[0].split("_")[0]

        for lbl in lbl_list:
            raw_data = pd.read_csv(root + "/" + filename, header=None)
            raw_data = raw_data[raw_data[25].isin([lbl])]
            raw_data = raw_data.reset_index(drop=True)

            split_index = raw_data.shape[0]/3 * 2 - 1

            train = train.append(raw_data.loc[:split_index, :])
            test = test.append(raw_data.loc[split_index+1:, :])

        train.to_csv("../data/txt_files/per_user_train_test/" + new_name + "_train.txt", index=False, header=False)
        test.to_csv("../data/txt_files/per_user_train_test/" + new_name + "_test.txt", index=False, header=False)


filelist = [ f for f in os.listdir("../data/txt_files/all_users_train_test/")]
for f in filelist:
    os.remove(os.path.join("../data/txt_files/all_users_train_test/", f))

# #combine all train and test files and create two files
# train = pd.DataFrame()
# test = pd.DataFrame()
# for root, dirs, files in os.walk("../data/txt_files/per_user_train_test/"):
#     for i, filename in enumerate(files):
#
#         if filename.startswith("."):
#             continue
#
#         data = pd.read_csv(root + "/" + filename, header=None)
#         if 'test' in filename:
#             test = test.append(data)
#         else:
#             train = train.append(data)
#
# train.to_csv("../data/txt_files/all_users_train_test/all_train.txt", index=False, header=False)
# test.to_csv("../data/txt_files/all_users_train_test/all_test.txt", index=False, header=False)


# #create arff files from txt files
#
# f = open("../data/header.txt", "r")
# header = str(f.read())
#
# for root, dirs, files in os.walk("../data/txt_files/all_users_train_test/"):
#     for i, filename in enumerate(files):
#
#         if filename.startswith("."):
#             continue
#
#         f_name = "../data/arff_files/all_users/" + filename.split(".")[0] + ".arff"
#         f = open(root + "/" + filename, 'r')
#         content = str(f.read())
#
#         appened_to_file(f_name, header)
#         appened_to_file(f_name, content)
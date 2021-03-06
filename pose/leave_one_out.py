import os
from math import acos

import numpy as np
import pandas as pd
import skinematics.quat as qt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import normalize
from sklearn.svm import SVC

lbl_list1 = ['knife_cutting', 'hammer_hammering', 'pen_writing', 'mug_drinking', 'spoon_stirring', 'none', 'screwdriver_screwing(no-release)', 'screwdriver_screwing(release)',
            'saw_sawing', 'knife_chopping', 'bottle_drinking']

lbl_list2 = ['knife_cutting', 'hammer_hammering', 'pen_writing', 'mug_drinking', 'spoon_stirring', 'none', 'screwdriver_screwing(no-release)',
            'saw_sawing', 'knife_chopping', 'bottle_drinking']

lbl_list3 = ['knife_cutting', 'hammer_hammering', 'pen_writing', 'mug_drinking', 'spoon_stirring', 'none', 'screwdriver_screwing(release)',
            'saw_sawing', 'knife_chopping', 'bottle_drinking']

full_list = [lbl_list2]


names_list = ['chamod' ]


def appened_to_file(flie_name, line, direct):
    with open(flie_name, "a") as write_file:
        if not direct:
            line = line.replace("[", "").replace("]", "").replace('\'', '')
        write_file.write(line + "\n")


def calculate_all_angles(data_frame, file):

    back = normalize(data_frame.loc[:, 0:3])
    pinky = normalize(data_frame.loc[:, 4:7])
    ring = normalize(data_frame.loc[:, 8:11])
    middl = normalize(data_frame.loc[:, 12:15])
    index = normalize(data_frame.loc[:, 16:19])
    thumb = normalize(data_frame.loc[:, 20:23])

    rel_p = normalize(qt.q_mult(pinky, qt.q_inv(back)))
    rel_r = normalize(qt.q_mult(ring, qt.q_inv(back)))
    rel_m = normalize(qt.q_mult(middl, qt.q_inv(back)))
    rel_i = normalize(qt.q_mult(index, qt.q_inv(back)))
    rel_t = normalize(qt.q_mult(thumb, qt.q_inv(back)))

    rel_0 = normalize(qt.q_mult(qt.q_inv(index), thumb))
    rel_1 = normalize(qt.q_mult(qt.q_inv(middl), thumb))
    rel_2 = normalize(qt.q_mult(qt.q_inv(ring), thumb))
    rel_3 = normalize(qt.q_mult(qt.q_inv(pinky), thumb))
    rel_4 = normalize(qt.q_mult(qt.q_inv(middl), index))
    rel_5 = normalize(qt.q_mult(qt.q_inv(ring), index))
    rel_6 = normalize(qt.q_mult(qt.q_inv(pinky), index))
    rel_7 = normalize(qt.q_mult(qt.q_inv(ring), middl))
    rel_8 = normalize(qt.q_mult(qt.q_inv(pinky), middl))
    rel_9 = normalize(qt.q_mult(qt.q_inv(pinky), ring))

    obj = data_frame.loc[:, 24]
    obj = obj.replace(['knife_cutting', 'knife_chopping'], 'knife')
    obj = np.array(obj)

    strng = ""
    for i in range(rel_p.shape[0]):
        strng = ""
        ang = np.rad2deg(2 * acos(rel_p[i, 0]))
        if ang > 180: ang = 360 - ang
        strng = str(ang) + ","
        ang = np.rad2deg(2 * acos(rel_r[i, 0]))
        if ang > 180: ang = 360 - ang
        strng = strng + str(ang) + ","
        ang = np.rad2deg(2 * acos(rel_m[i, 0]))
        if ang > 180: ang = 360 - ang
        strng = strng + str(ang) + ","
        ang = np.rad2deg(2 * acos(rel_i[i, 0]))
        if ang > 180: ang = 360 - ang
        strng = strng + str(ang) + ","
        ang = np.rad2deg(2 * acos(rel_t[i, 0]))
        if ang > 180: ang = 360 - ang
        strng = strng + str(ang) + ","

        ang = np.rad2deg(2 * acos(rel_0[i, 0]))
        if ang > 180: ang = 360 - ang
        strng = strng + str(ang) + ","
        ang = np.rad2deg(2 * acos(rel_1[i, 0]))
        if ang > 180: ang = 360 - ang
        strng = strng + str(ang) + ","
        ang = np.rad2deg(2 * acos(rel_2[i, 0]))
        if ang > 180: ang = 360 - ang
        strng = strng + str(ang) + ","
        ang = np.rad2deg(2 * acos(rel_3[i, 0]))
        if ang > 180: ang = 360 - ang
        strng = strng + str(ang) + ","
        ang = np.rad2deg(2 * acos(rel_4[i, 0]))
        if ang > 180: ang = 360 - ang
        strng = strng + str(ang) + ","
        ang = np.rad2deg(2 * acos(rel_5[i, 0]))
        if ang > 180: ang = 360 - ang
        strng = strng + str(ang) + ","
        ang = np.rad2deg(2 * acos(rel_6[i, 0]))
        if ang > 180: ang = 360 - ang
        strng = strng + str(ang) + ","
        ang = np.rad2deg(2 * acos(rel_7[i, 0]))
        if ang > 180: ang = 360 - ang
        strng = strng + str(ang) + ","
        ang = np.rad2deg(2 * acos(rel_8[i, 0]))
        if ang > 180: ang = 360 - ang
        strng = strng + str(ang) + ","
        ang = np.rad2deg(2 * acos(rel_9[i, 0]))
        if ang > 180: ang = 360 - ang
        strng = strng + str(ang) + ","
        strng = strng + obj[i]

        appened_to_file(file, strng, False)


def process_file(file, target_file, lbl_list):

    data = pd.DataFrame()
    print("\n\n" + target_file + "\n")

    for lbl in lbl_list:
        raw_data = pd.read_csv(file, header=None)
        raw_data = raw_data[raw_data[24].isin([lbl])]
        raw_data = raw_data.reset_index(drop=True)

        if 'none' not in lbl:
            raw_data = raw_data.loc[:raw_data.shape[0] / 2 - 1, :]
            raw_data = raw_data.reset_index(drop=True)

        if 'knife' in lbl:
            raw_data = raw_data.loc[:raw_data.shape[0]/2-1, :]

        data = data.append(raw_data)
        data = data.reset_index(drop=True)

    print("data set size: " + str(data.groupby([24]).size()))
    calculate_all_angles(data, "../data/pose/user/" + target_file + ".txt")


def run_machine_learning(train_df, test_df):

    x_test = test_df.loc[:, :14]
    y_test = test_df.loc[:, 15]

    x_train = train_df.loc[:, :14]
    y_train = train_df.loc[:, 15]

    x_combined = x_train.append(x_test)
    x_combined = (x_combined - x_combined.mean()) / x_combined.std() + 0.000001
    x_combined = x_combined.reset_index(drop=True)

    x_train = x_combined.loc[:x_train.shape[0] - 1, :]
    x_test = x_combined.loc[x_train.shape[0]:, :]

    svm = SVC()
    svm.fit(x_train, y_train)
    y_pred=svm.predict(x_test)
    score = accuracy_score(y_test, y_pred)
    print('\nSVM score: ' + str(score))
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print('\n------------------------------------\n\n')

    rf = RandomForestClassifier()
    rf.fit(x_train, y_train)
    y_pred=rf.predict(x_test)
    score = accuracy_score(y_test, y_pred)
    print('\nRF score: ' + str(score))
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print('\n------------------------------------\n\n')

    nn_score = 0.0
    report = ''
    conf_matx = ''
    for i in range(8):
        clf = MLPClassifier(activation='relu', hidden_layer_sizes=(200, 200), max_iter=200000, learning_rate='adaptive', early_stopping=False)
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        score = accuracy_score(y_test, y_pred)
        if score > nn_score:
            nn_score = score
            report = classification_report(y_test, y_pred)
            conf_matx = confusion_matrix(y_test, y_pred)


    print('\nNN score: ' + str(nn_score))
    print(report)
    print(conf_matx)
    print('\n------------------------------------\n\n')


def setup_data(lbl_list):

    filelist = [f for f in os.listdir("../data/pose/user/")]
    for f in filelist:
        os.remove(os.path.join("../data/pose/user/", f))

    print("\n\n************************* start data processing... ************************* \n\n")

    for root, dirs, files in os.walk("/Users/thisum/Documents/Personel_Docs/NUS_MSc/Research/MagicHand_Data/posture_data/raw/"):
        for i, filename in enumerate(files):

            if filename.startswith("."):
                continue

            process_file(root + "/" + filename, filename.split(".")[0], lbl_list)

# print("\n\n************************* data processing is done... ************************* \n\n")


def evaluate_algorithms():

    for name in names_list:

        train = pd.DataFrame()
        test = pd.DataFrame()

        for root, dirs, files in os.walk("../data/pose/user/"):
            for i, filename in enumerate(files):

                if filename.startswith("."):
                    continue

                data = pd.read_csv(root + "/" + filename, header=None)

                if name in filename:
                    test = test.append(data)
                else:
                    train = train.append(data)

        test = test.reset_index(drop=True)
        train = train.reset_index(drop=True)

        print("\n ******************** " + name + " ******************** \n")
        print("train size: " + str(train.shape) + "    test size: " + str(test.shape))
        run_machine_learning(train, test)

        print("\n ******************** ******************** \n")


print("\n ******************** start walk up method ******************** \n")

for l in full_list:
    print("\n ******************** ********************")
    print("\nlist : " + str(l))
    setup_data(l)
    # evaluate_algorithms()

    print("\nend list\n\n")

import os
from math import acos

import pickle
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


names_list = ['chamod', 'issabelle', 'juanpa', 'julia', 'omkar', 'pablo', 'rivindu', 'samitha', 'thisum', 'vipula', 'yilei', 'yvonne' ]

activity_list = ['bottle_drinking', 'hammer_hammering', 'knife_chopping', 'knife_cutting', 'mug_drinking', 'none' , 'pen_writing', 'saw_sawing',
             'screwdriver_screwing(release)', 'spoon_stirring']

pose_list = ['bottle_drinking', 'hammer_hammering', 'knife_chopping', 'knife_cutting', 'mug_drinking' , 'none', 'pen_writing', 'saw_sawing',
             'screwdriver_screwing(no-release)',  'spoon_stirring']

activities = ['DB', 'HA', 'CH', 'CK', 'DM', 'RE', 'WR', 'SA', 'SC', 'ST']
postures = ['DB', 'HA', 'CH', 'CK', 'DM', 'RE', 'WR', 'SA', 'SC', 'ST']


def print_confusion_matrix(list):
    print("\n")
    for i in list:
        print(str(i).replace("]", '').replace('[', ''))

    print("\n\n")

def evaluate_algorithms(fold, name, activity_test, pose_test):

    activity_model = None
    pkl_filename = "../data/ensemble/models/activity_" + name + "_" + str(fold) + ".pkl"
    with open(pkl_filename, 'rb') as file:
        activity_model = pickle.load(file)

    pose_model = None
    pkl_filename = "../data/ensemble/models/posture_" + name + "_" + str(fold) + ".pkl"
    with open(pkl_filename, 'rb') as file:
        pose_model = pickle.load(file)

    print("**********************************\n")
    print(name)

    List = []

    for i in range(10):
        s = pd.Series([0,0,0,0,0,0,0,0,0,0], index=activities)

        p_test = pose_test[pose_test[15].isin([pose_list[i]])]
        p_test = p_test.iloc[99:, :]
        p_test = p_test.reset_index(drop=True)
        a_test = activity_test[activity_test[25].isin([activity_list[i]])]
        a_test = a_test.reset_index(drop=True)

        p_x_test = p_test.loc[:, :14]
        p_y_test = p_test.loc[:, 15]

        a_x_test = a_test.loc[:, :24]
        a_y_test = a_test.loc[:, 25]

        a_nn_score = 0.0
        p_nn_score = 0.0
        report = ''
        conf_matx = ''
        for i in range(5):

            p_y_pred = pose_model.predict(p_x_test)
            a_y_pred = activity_model.predict(a_x_test)
            p_score = accuracy_score(p_y_test, p_y_pred)
            a_score = accuracy_score(a_y_test, a_y_pred)

            if p_score > p_nn_score:
                p_nn_score = p_score
                p = pose_model.predict_proba(p_x_test)

            if a_score > a_nn_score:
                a_nn_score = a_score
                a = activity_model.predict_proba(a_x_test)

        results = []
        result_df = pd.DataFrame(columns=['activity', 'posture', 'ensemble'])

        size = 0
        if p.shape[0] > a.shape[0]:
            size = a.shape[0]
        else:
            size = p.shape[0]

        for r in range(size):

            activity_accuracy = []
            posture_accuracy = []
            prediction = ''

            for i in range(p[r].shape[0]):
                posture_accuracy.append((postures[i], p[r][i]))

            for i in range(a[r].shape[0]):
                activity_accuracy.append((activities[i], a[r][i]))

            max_activity = sorted(activity_accuracy, key=lambda x: x[1], reverse=True)[:3]
            max_posture = sorted(posture_accuracy, key=lambda x: x[1], reverse=True)[:3]

            if max_posture[0][0] in (['DB', 'DM', 'WR']) and max_posture[0][1] > 0.95:
                prediction = max_posture[0][0]

            elif max_posture[0][1] > max_activity[0][1] and max_activity[0][1] - max_activity[1][1] < 0.9:
                prediction = max_posture[0][0]
            else:
                prediction = max_activity[0][0]
                # if max_activity[0][1] - max_activity[1][1] > 0.8:
                #     prediction = max_activity[0][0]
                # elif max_posture[0][1] > 0.5:
                #     prediction = max_posture[0][0]
                # else:
                #     prediction = max_activity[0][0]

            result_df = result_df.append(
                {'activity': max_activity[0][0], 'posture': max_posture[0][0], 'ensemble': prediction}, ignore_index=True)

        g = result_df.groupby(['ensemble']).size()
        for i, v in g.items():
            s[i] = v

        List.append(s.values.tolist())

    print_confusion_matrix(List)

def load_activity_files(name):

    for root, dirs, files in os.walk("../data/activity/txt_files/per_user_train_test/"):
        for i, filename in enumerate(files):

            if filename.startswith("."):
                continue

            if name in filename and 'test' in filename:
                test = pd.read_csv(root + "/" + filename, header=None)
            elif name in filename and 'train' in filename:
                train = pd.read_csv(root + "/" + filename, header=None)

    x_test = test.loc[:,:24]
    y_test = test.loc[:,25]

    x_train = train.loc[:,:24]
    y_train = train.loc[:,25]

    x_combined = x_train.append(x_test)
    x_combined =(x_combined-x_combined.mean())/x_combined.std()+0.000001
    x_combined = x_combined.reset_index(drop=True)

    x_test = x_combined.loc[x_train.shape[0]:, :]

    a_x_test = x_test
    a_y_test = y_test

    return (a_x_test, a_y_test)


def load_pose_files(name):

    for root, dirs, files in os.walk("../data/pose/per_user/"):
        for i, filename in enumerate(files):

            if filename.startswith("."):
                continue

            if name in filename and 'test' in filename:
                test = pd.read_csv(root + "/" + filename, header=None)
            elif name in filename and 'train' in filename:
                train = pd.read_csv(root + "/" + filename, header=None)

    train_df = train.sample(frac=1).reset_index(drop=True)
    test_df = test.sample(frac=1).reset_index(drop=True)

    x_test = test_df.loc[:, :14]
    y_test = test_df.loc[:, 15]

    x_train = train_df.loc[:, :14]
    y_train = train_df.loc[:, 15]

    x_combined = x_train.append(x_test)
    x_combined = (x_combined - x_combined.mean()) / x_combined.std() + 0.000001
    x_combined = x_combined.reset_index(drop=True)

    p_x_test = x_combined.loc[x_train.shape[0]:, :]
    p_y_test = y_test

    return (p_x_test, p_y_test)



def load_files(name):

    (a_x_test, a_y_test) = load_activity_files(name)
    (p_x_test, p_y_test) = load_pose_files(name)

    dat1 = a_x_test.reset_index(drop=True)
    dat2 = a_y_test.reset_index(drop=True)
    activity_test = pd.concat([dat1, dat2], axis=1)

    dat1 = p_x_test.reset_index(drop=True)
    dat2 = p_y_test.reset_index(drop=True)
    pose_test = pd.concat([dat1, dat2], axis=1)

    return (activity_test, pose_test)

def evaluate_ensemble():

    for name in names_list:
        (activity_test, pose_test) = load_files(name)
        evaluate_algorithms(2, name, activity_test, pose_test)


evaluate_ensemble()
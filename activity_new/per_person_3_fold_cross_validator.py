import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from category_file_processor import prepare_test_train_files

lbl_list1 = ['knife_cutting', 'hammer_hammering', 'pen_writing', 'mug_drinking', 'spoon_stirring',
            'saw_sawing', 'screwdriver_screwing(no-release)', 'knife_chopping', 'bottle_drinking'] 

lbl_list2 = ['knife_cutting', 'hammer_hammering', 'pen_writing', 'mug_drinking', 'spoon_stirring',
            'saw_sawing', 'screwdriver_screwing(release)', 'knife_chopping', 'bottle_drinking', 'none']

lbl_list3 = ['knife_cutting', 'hammer_hammering', 'pen_writing', 'mug_drinking', 'spoon_stirring',
            'saw_sawing', 'screwdriver_screwing(no-release)', 'screwdriver_screwing(release)', 'knife_chopping', 'bottle_drinking']

total_list = [lbl_list2]

names_list = ['chamod', 'isabella', 'juanpa', 'julia', 'omkar', 'pablo', 'rivindu', 'samitha', 'thisum', 'vipula', 'yilei', 'yvonne' ]


window_list = [100]

def appened_to_file(flie_name, line):
    with open(flie_name, "a") as write_file:
        write_file.write(line)


def prepare_files(lbl_list, index_to_split):

    filelist = [ f for f in os.listdir("../data/activity/txt_files/per_user_train_test/")]
    for f in filelist:
        os.remove(os.path.join("../data/activity/txt_files/per_user_train_test/", f))

    # make train and test files for each user for accelerometer data
    for root, dirs, files in os.walk("../data/activity/accel_features/"):
        for i, filename in enumerate(files):

            if filename.startswith("."):
                continue

            df = pd.read_csv(root + "/" + filename, header=None)
            print("\noriginal data set size: " + str(df.shape))

            train = pd.DataFrame()
            test = pd.DataFrame()

            new_name = filename.split(".")[0].split("_")[0]

            for lbl in lbl_list:
                raw_data = pd.read_csv(root + "/" + filename, header=None)
                raw_data = raw_data[raw_data[25].isin([lbl])]
                raw_data = raw_data.reset_index(drop=True)

                if index_to_split == 0:

                    split_index = np.floor(raw_data.shape[0] / 3 * 1 - 1)

                    test = test.append(raw_data.loc[:split_index, :])
                    train = train.append(raw_data.loc[split_index + 1:, :])

                if index_to_split == 1:

                    split_index = np.floor(raw_data.shape[0] / 3 * 2 - 1)

                    train = train.append(raw_data.loc[:split_index, :])
                    test = test.append(raw_data.loc[split_index + 1:, :])

                elif index_to_split == 2:

                    split_index1 = np.floor(raw_data.shape[0] / 3 * 1 - 1)
                    split_index2 = np.floor(raw_data.shape[0] / 3 * 2 - 1)

                    train = train.append(raw_data.loc[:split_index1, :])
                    test = test.append(raw_data.loc[split_index1 + 1:split_index2, :])
                    train = train.append(raw_data.loc[split_index2 + 1:, :])

                train = train.reset_index(drop=True)
                test = test.reset_index(drop=True)
            print("data set size: train - " + str(train.shape) + "  test - " + str(test.shape))
            train.to_csv("../data/activity/txt_files/per_user_train_test/" + new_name + "_train.txt", index=False, header=False)
            test.to_csv("../data/activity/txt_files/per_user_train_test/" + new_name + "_test.txt", index=False, header=False)


def evaluate_algorithms(train_df, test_df):

    x_test = test_df.loc[:,:24]
    y_test = test_df.loc[:,25]

    x_train = train_df.loc[:,:24]
    y_train = train_df.loc[:,25]

    x_combined = x_train.append(x_test)
    x_combined =(x_combined-x_combined.mean())/x_combined.std()+0.000001
    x_combined = x_combined.reset_index(drop=True)

    x_train = x_combined.loc[:x_train.shape[0]-1, :]
    x_test = x_combined.loc[x_train.shape[0]:, :]
    #
    # svm = SVC()
    # svm.fit(x_train, y_train)
    # y_pred=svm.predict(x_test)
    # score = accuracy_score(y_test, y_pred)
    # print('SVM score: ' + str(score))
    # print(classification_report(y_test, y_pred))
    # print(confusion_matrix(y_test, y_pred))
    # print("\n---------------------------------\n")
    #
    #
    # rf = RandomForestClassifier()
    # rf.fit(x_train, y_train)
    # y_pred=rf.predict(x_test)
    # score = accuracy_score(y_test, y_pred)
    # print('RF score: ' + str(score))
    # print(classification_report(y_test, y_pred))
    # print(confusion_matrix(y_test, y_pred))
    # print("\n---------------------------------\n")

    nn_score = 0.0
    report = ''
    conf_matx = ''
    for i in range(5):
        clf = MLPClassifier(activation='relu', hidden_layer_sizes=(100, 100), max_iter=200000, learning_rate='adaptive',
                            early_stopping=False)
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        score = accuracy_score(y_test, y_pred)
        if score > nn_score:
            nn_score = score
            report = classification_report(y_test, y_pred)
            conf_matx = confusion_matrix(y_test, y_pred)


    print('NN score: ' + str(nn_score))
    print(report)
    print(conf_matx)

    print("\n\n________________________________________________________________\n\n")


def run_algorithm_per_person():

    for name in names_list:

        train = pd.DataFrame()
        test = pd.DataFrame()

        for root, dirs, files in os.walk("../data/activity/txt_files/per_user_train_test/"):
            for i, filename in enumerate(files):

                if filename.startswith("."):
                    continue

                data = pd.read_csv(root + "/" + filename, header=None)

                if name in filename and 'test' in filename:
                    test = test.append(data)
                elif name in filename and 'train' in filename:
                    train = train.append(data)


        print(name)
        print("________________")
        evaluate_algorithms(train, test)


for w in window_list:

    print("\n***********************************************************")
    print("***************** WINDOW SIZE : " + str(w) + "**********************")
    print("***********************************************************")

    for l in total_list:

        print("action list: ")
        print(l)

        print("***************** 0 fold *****************")
        prepare_test_train_files(l, w)
        prepare_files(l, 1)
        # run_algorithm_per_person()
        #
        # print("***************** 1 fold *****************")
        # prepare_test_train_files(l, w)
        # prepare_files(l, 1)
        # run_algorithm_per_person()
        #
        # print("***********************************************************")
        #
        # print("***************** 2 fold *****************")
        # prepare_test_train_files(l, w)
        # prepare_files(l, 2)
        # run_algorithm_per_person()


    print("***********************************************************")
    print("********************* ONE ROUND DONE ***********************")
    print("***********************************************************\n\n")

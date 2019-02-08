import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy import fftpack
import os
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

# #evaluate for all results of activity
# test_df = pd.read_csv("../data/txt_files/all_users_train_test/all_test.txt", header=None)
# train_df = pd.read_csv("../data/txt_files/all_users_train_test/all_train.txt", header=None)
#
# x_test = test_df.loc[:,:24]
# y_test = test_df.loc[:,25]
#
# x_train = train_df.loc[:,:24]
# y_train = train_df.loc[:,25]
#
# x_combined = x_train.append(x_test)
# x_combined =(x_combined-x_combined.mean())/x_combined.std()+0.000001
# x_combined = x_combined.reset_index(drop=True)
#
# x_train = x_combined.loc[:x_train.shape[0]-1, :]
# x_test = x_combined.loc[x_train.shape[0]:, :]
#
#
# svm = SVC()
# svm.fit(x_train, y_train)
# y_pred=svm.predict(x_test)
# score = accuracy_score(y_test, y_pred)
# print('SVM score: ' + str(score))
# print(classification_report(y_test, y_pred))
# print("\n---------------------------------\n")
#
#
# rf = RandomForestClassifier()
# rf.fit(x_train, y_train)
# y_pred=rf.predict(x_test)
# score = accuracy_score(y_test, y_pred)
# print('RF score: ' + str(score))
# print(classification_report(y_test, y_pred))
# print("\n---------------------------------\n")
#
# nn_score = 0.0
# report = ''
# conf_matx = ''
# for i in range(10):
#     clf = MLPClassifier(activation='relu', hidden_layer_sizes=(100, 100), max_iter=200000, learning_rate='adaptive',
#                         early_stopping=False)
#     clf.fit(x_train, y_train)
#     y_pred = clf.predict(x_test)
#     score = accuracy_score(y_test, y_pred)
#     if score > nn_score:
#         nn_score = score
#         report = classification_report(y_test, y_pred)
#         conf_matx = confusion_matrix(y_test, y_pred)
#
#
# print('NN score: ' + str(nn_score))
# print(report)
# print(conf_matx)


#leave one out validation method
names_list = ['chamod', 'isabella', 'juanpa', 'julia', 'omkar', 'pablo', 'rivindu', 'samitha', 'thisum', 'vipula', 'yilei', 'yvonne' ]

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

    svm = SVC()
    svm.fit(x_train, y_train)
    y_pred=svm.predict(x_test)
    score = accuracy_score(y_test, y_pred)
    print('SVM score: ' + str(score))
    print(classification_report(y_test, y_pred))
    print("\n---------------------------------\n")


    rf = RandomForestClassifier()
    rf.fit(x_train, y_train)
    y_pred=rf.predict(x_test)
    score = accuracy_score(y_test, y_pred)
    print('RF score: ' + str(score))
    print(classification_report(y_test, y_pred))
    print("\n---------------------------------\n")

    nn_score = 0.0
    report = ''
    conf_matx = ''
    for i in range(8):
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


for name in names_list:

    train = pd.DataFrame()
    test = pd.DataFrame()

    for root, dirs, files in os.walk("../data/txt_files/per_user_train_test/"):
        for i, filename in enumerate(files):

            if filename.startswith("."):
                continue

            data = pd.read_csv(root + "/" + filename, header=None)

            if name in filename:
                test = test.append(data)
            else:
                train = train.append(data)


    print(name)
    print("________________")
    evaluate_algorithms(train, test)



# #per person validation method
# names_list = ['chamod', 'isabella', 'juanpa', 'julia', 'omkar', 'pablo', 'rivindu', 'samitha', 'thisum', 'vipula', 'yilei', 'yvonne' ]
#
# def evaluate_algorithms(train_df, test_df):
#
#     x_test = test_df.loc[:,:24]
#     y_test = test_df.loc[:,25]
#
#     x_train = train_df.loc[:,:24]
#     y_train = train_df.loc[:,25]
#
#     x_combined = x_train.append(x_test)
#     x_combined =(x_combined-x_combined.mean())/x_combined.std()+0.000001
#     x_combined = x_combined.reset_index(drop=True)
#
#     x_train = x_combined.loc[:x_train.shape[0]-1, :]
#     x_test = x_combined.loc[x_train.shape[0]:, :]
#
#     svm = SVC()
#     svm.fit(x_train, y_train)
#     y_pred=svm.predict(x_test)
#     score = accuracy_score(y_test, y_pred)
#     print('SVM score: ' + str(score))
#     print(classification_report(y_test, y_pred))
#     print("\n---------------------------------\n")
#
#
#     rf = RandomForestClassifier()
#     rf.fit(x_train, y_train)
#     y_pred=rf.predict(x_test)
#     score = accuracy_score(y_test, y_pred)
#     print('RF score: ' + str(score))
#     print(classification_report(y_test, y_pred))
#     print("\n---------------------------------\n")
#
#     nn_score = 0.0
#     report = ''
#     conf_matx = ''
#     for i in range(10):
#         clf = MLPClassifier(activation='relu', hidden_layer_sizes=(100, 100), max_iter=200000, learning_rate='adaptive',
#                             early_stopping=False)
#         clf.fit(x_train, y_train)
#         y_pred = clf.predict(x_test)
#         score = accuracy_score(y_test, y_pred)
#         if score > nn_score:
#             nn_score = score
#             report = classification_report(y_test, y_pred)
#             conf_matx = confusion_matrix(y_test, y_pred)
#
#
#     print('NN score: ' + str(nn_score))
#     print(report)
#     print(conf_matx)
#
#     print("\n\n________________________________________________________________\n\n")
#
#
# for name in names_list:
#
#     train = pd.DataFrame()
#     test = pd.DataFrame()
#
#     for root, dirs, files in os.walk("../data/txt_files/per_user_train_test/"):
#         for i, filename in enumerate(files):
#
#             if filename.startswith("."):
#                 continue
#
#             data = pd.read_csv(root + "/" + filename, header=None)
#
#             if name in filename and 'test' in filename:
#                 test = test.append(data)
#             elif name in filename and 'train' in filename:
#                 train = train.append(data)
#
#
#     print(name)
#     print("________________")
#     evaluate_algorithms(train, test)
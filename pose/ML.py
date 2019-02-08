import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.grid_search import GridSearchCV

# from sklearn.metrics import accuracy_score
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import accuracy_score
# from sklearn.neural_network import MLPClassifier
# from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
# from sklearn.svm import SVC
# from sklearn.svm import SVC
# from skinematics.quat import calc_quat
from sklearn.preprocessing import normalize
import skinematics.quat as qt
from math import atan2
from math import acos
import matplotlib.pyplot as plt




# data_frame=pd.read_csv("/Users/thisum/Documents/AppDevelopment/Python/GoD/angle_train.arff", header=None)
# train_x = data_frame.loc[:,:14]
# train_y = data_frame.loc[:,15]
#
# data_frame=pd.read_csv("/Users/thisum/Documents/AppDevelopment/Python/GoD/angle_test.arff", header=None)
# test_x = data_frame.loc[:,:14]
# test_y = data_frame.loc[:,15]
#
# svm = SVC()
# svm.fit(train_x, train_y)
# y_pred=svm.predict(test_x)
# score = accuracy_score(test_y, y_pred)
# print('SVM score: ' + str(score))
#
# clf = MLPClassifier(activation='relu', hidden_layer_sizes=(20,20), max_iter=200000, learning_rate='adaptive', early_stopping=False)
# clf.fit(train_x, train_y)
# y_pred=clf.predict(test_x)
# score = accuracy_score(test_y, y_pred)
# print('NN score: ' + str(score))
#
# rf = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
# rf.fit(train_x, train_y)
# y_pred=rf.predict(test_x)
# score = accuracy_score(test_y, y_pred)
# print('RF score: ' + str(score))
# print(confusion_matrix(test_y, y_pred))

"""
 With two sensors
"""

# angles = []
# data_frame=pd.read_csv("/Users/thisum/Desktop/relax.txt", header=None)
#
# A = normalize(data_frame.loc[:, 0:3])
# B = normalize(data_frame.loc[:, 4:7])
#
# quatDiff = qt.q_mult(qt.q_conj(B), A)
#
# p = quatDiff[:, 1:4]
# q = np.sqrt(np.sum(np.multiply(p,p), axis=1))
#
# for i in range(p.shape[0]):
#     ang = 2*atan2(q[i],quatDiff[i,0])
#     ang = np.rad2deg(ang)
#     if ang > 180:
#         ang = 360 - ang
#     print(ang)
#     angles.append(np.rad2deg(ang))
#
# print(angles)
# plt.hist(angles, bins='auto', color='#0504aa',alpha=0.7, rwidth=0.85)
# plt.show()


#
# angles = []
# data_frame=pd.read_csv("/Users/thisum/Desktop/test_joint.txt", header=None)
#
# A = normalize(data_frame.loc[:, 0:3])
# B = normalize(data_frame.loc[:, 4:7])
#
# mul = B * qt.q_inv(A)
# for i in range(mul.shape[0]):
#     ang = acos(mul[i,0])
#     angles.append(np.rad2deg(ang))
#
# plt.hist(angles, bins='auto', color='#0504aa',alpha=0.7, rwidth=0.85)
# plt.show()

"""
 With all 6 sensors
"""

# data_frame=pd.read_csv("/Users/thisum/Desktop/quat_full.txt", header=None)
#
# back = normalize(data_frame.loc[:, 0:3])
# A = normalize(data_frame.loc[:, 4:7])
# B = normalize(data_frame.loc[:, 8:11])
# C = normalize(data_frame.loc[:, 12:15])
# D = normalize(data_frame.loc[:, 16:19])
# E = normalize(data_frame.loc[:, 20:23])
# F = data_frame.loc[:, 24]
#
# quatDiffA = qt.q_mult(qt.q_conj(back), A)
# quatDiffB = qt.q_mult(qt.q_conj(back), B)
# quatDiffC = qt.q_mult(qt.q_conj(back), C)
# quatDiffD = qt.q_mult(qt.q_conj(back), D)
# quatDiffE = qt.q_mult(qt.q_conj(back), E)
#
# p_A = quatDiffA[:, 1:4]
# p_B = quatDiffB[:, 1:4]
# p_C = quatDiffC[:, 1:4]
# p_D = quatDiffD[:, 1:4]
# p_E = quatDiffE[:, 1:4]
#
# q_A = np.sqrt(np.sum(np.multiply(p_A,p_A), axis=1))
# q_B = np.sqrt(np.sum(np.multiply(p_B,p_B), axis=1))
# q_C = np.sqrt(np.sum(np.multiply(p_C,p_C), axis=1))
# q_D = np.sqrt(np.sum(np.multiply(p_D,p_D), axis=1))
# q_E = np.sqrt(np.sum(np.multiply(p_E,p_E), axis=1))
#
# for i in range(q_A.size):
#     ang_A = 2*atan2(q_A[i],quatDiffA[i,0])
#     ang_B = 2*atan2(q_B[i],quatDiffB[i,0])
#     ang_C = 2*atan2(q_C[i],quatDiffC[i,0])
#     ang_D = 2*atan2(q_D[i],quatDiffD[i,0])
#     ang_E = 2*atan2(q_E[i],quatDiffE[i,0])
#     print(str(np.rad2deg(ang_A)) + "," + str(np.rad2deg(ang_B)) + "," + str(np.rad2deg(ang_C)) + "," + str(np.rad2deg(ang_D)) + "," + str(np.rad2deg(ang_E) ) + "," + str(F[i]))


"""
 considering the initial positions
"""
#
q_b0 = np.array([0.9862, -0.0078, 0.0343, 0.1618, 0.9845, -0.0107, 0.0341, 0.1715, 0.9982, -6.0E-4, 0.0441, -0.0402, 0.9959, 0.0213, -0.0101, -0.0871, 0.9852, 0.026, 0.0216, -0.1679, 0.0679, 0.0069, -0.0181, 0.9975 ])
q_f0 = np.array([0.9776, 0.0035, -0.0007, -0.2106])
#
angles = []
data_frame=pd.read_csv("/Users/thisum/Desktop/test.txt", header=None)

back_hand = normalize(data_frame.loc[:, 0:3])
finger_tip = normalize(data_frame.loc[:, 4:7])

finger = qt.q_mult(finger_tip , qt.q_inv(q_f0))
back = qt.q_mult(back_hand , qt.q_inv(q_b0))

rel = qt.q_mult(finger_tip, qt.q_inv(back_hand))
rel = normalize(rel)

for i in range(rel.shape[0]):
    rad = 2*acos(rel[i,0])
    ang = np.rad2deg(rad)
    if ang > 180: ang = 360-ang
    print(ang)
    angles.append(ang)
plt.hist(angles, bins='auto', color='#0504aa',alpha=0.7, rwidth=0.85)
plt.show()


# p = rel[:, 1:4]
# q = np.sqrt(np.sum(np.multiply(p,p), axis=1))
#
# for i in range(p.shape[0]):
#     ang = 2*atan2(q[i],rel[i,0])
#     print(np.rad2deg(ang))






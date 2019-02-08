from math import acos
import numpy as np
import pandas as pd
import skinematics.quat as qt
from sklearn.preprocessing import normalize


#

def appened_to_file(flie_name, str):

    with open(flie_name, "a") as write_file:
        write_file.write(str)

data_frame = pd.read_csv("/Users/thisum/Desktop/MagicHand_Data/sequence_data/quart_test.csv", header=None)

# writing quaternions with respect to the back of the hand
obj = data_frame.loc[:, 24]
back = normalize(data_frame.loc[:, 0:3])
pinky = normalize(data_frame.loc[:, 4:7])
ring = normalize(data_frame.loc[:, 8:11])
middl = normalize(data_frame.loc[:, 12:15])
index = normalize(data_frame.loc[:, 16:19])
thumb = normalize(data_frame.loc[:, 20:23])

rel_p = normalize(qt.q_mult(pinky, qt.q_inv(back)))
rel_r = normalize(qt.q_mult(ring , qt.q_inv(back)))
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
    strng = strng + obj[i] + "\n"
    # print(strng)
    appened_to_file("quat_test.arff", strng)



# writing quaternions with relative to it's initial positions and then to the back of the hand


# q_b0 = np.array([0.9862, -0.0078, 0.0343, 0.1618])
# q_p0 = np.array([0.9845, -0.0107, 0.0341, 0.1715])
# q_r0 = np.array([0.9982, -6.0E-4, 0.0441, -0.0402])
# q_m0 = np.array([0.9959, 0.0213, -0.0101, -0.0871])
# q_i0 = np.array([0.9852, 0.026, 0.0216, -0.1679])
# q_t0 = np.array([0.0679, 0.0069, -0.0181, 0.9975])
#
#
#
#
#
# obj = data_frame.loc[:, 24]
# back  = normalize(data_frame.loc[:, 0:3])
# pinky = normalize(data_frame.loc[:, 4:7])
# ring  = normalize(data_frame.loc[:, 8:11])
# middl = normalize(data_frame.loc[:, 12:15])
# index = normalize(data_frame.loc[:, 16:19])
# thumb = normalize(data_frame.loc[:, 20:23])
#
# back  =  normalize(qt.q_mult(qt.q_inv(back), pinky))
# pinky =  normalize(qt.q_mult(qt.q_inv(back), ring))
# ring  =  normalize(qt.q_mult(qt.q_inv(back), middl))
# middl =  normalize(qt.q_mult(qt.q_inv(back), index))
# index =  normalize(qt.q_mult(qt.q_inv(back), thumb))
# thumb =  normalize(qt.q_mult(qt.q_inv(back), pinky))
#
# rel_p = normalize(qt.q_mult(qt.q_inv(back), pinky))
# rel_r = normalize(qt.q_mult(qt.q_inv(back), ring))
# rel_m = normalize(qt.q_mult(qt.q_inv(back), middl))
# rel_i = normalize(qt.q_mult(qt.q_inv(back), index))
# rel_t = normalize(qt.q_mult(qt.q_inv(back), thumb))
#
# rel_0 = normalize(qt.q_mult(qt.q_inv(index), thumb))
# rel_1 = normalize(qt.q_mult(qt.q_inv(middl), thumb))
# rel_2 = normalize(qt.q_mult(qt.q_inv(ring), thumb))
# rel_3 = normalize(qt.q_mult(qt.q_inv(pinky), thumb))
# rel_4 = normalize(qt.q_mult(qt.q_inv(middl), index))
# rel_5 = normalize(qt.q_mult(qt.q_inv(ring), index))
# rel_6 = normalize(qt.q_mult(qt.q_inv(pinky), index))
# rel_7 = normalize(qt.q_mult(qt.q_inv(ring), middl))
# rel_8 = normalize(qt.q_mult(qt.q_inv(pinky), middl))
# rel_9 = normalize(qt.q_mult(qt.q_inv(pinky), ring))
#
# strng = ""
# for i in range(rel_p.shape[0]):
#     strng = ""
#     ang = np.rad2deg(2 * acos(rel_p[i, 0]))
#     if ang > 180: ang = 360 - ang
#     strng = str(ang) + ","
#     ang = np.rad2deg(2 * acos(rel_r[i, 0]))
#     if ang > 180: ang = 360 - ang
#     strng = strng + str(ang) + ","
#     ang = np.rad2deg(2 * acos(rel_m[i, 0]))
#     if ang > 180: ang = 360 - ang
#     strng = strng + str(ang) + ","
#     ang = np.rad2deg(2 * acos(rel_i[i, 0]))
#     if ang > 180: ang = 360 - ang
#     strng = strng + str(ang) + ","
#     ang = np.rad2deg(2 * acos(rel_t[i, 0]))
#     if ang > 180: ang = 360 - ang
#     strng = strng + str(ang) + ","
#
#     ang = np.rad2deg(2 * acos(rel_0[i, 0]))
#     if ang > 180: ang = 360 - ang
#     strng = strng + str(ang) + ","
#     ang = np.rad2deg(2 * acos(rel_1[i, 0]))
#     if ang > 180: ang = 360 - ang
#     strng = strng + str(ang) + ","
#     ang = np.rad2deg(2 * acos(rel_2[i, 0]))
#     if ang > 180: ang = 360 - ang
#     strng = strng + str(ang) + ","
#     ang = np.rad2deg(2 * acos(rel_3[i, 0]))
#     if ang > 180: ang = 360 - ang
#     strng = strng + str(ang) + ","
#     ang = np.rad2deg(2 * acos(rel_4[i, 0]))
#     if ang > 180: ang = 360 - ang
#     strng = strng + str(ang) + ","
#     ang = np.rad2deg(2 * acos(rel_5[i, 0]))
#     if ang > 180: ang = 360 - ang
#     strng = strng + str(ang) + ","
#     ang = np.rad2deg(2 * acos(rel_6[i, 0]))
#     if ang > 180: ang = 360 - ang
#     strng = strng + str(ang) + ","
#     ang = np.rad2deg(2 * acos(rel_7[i, 0]))
#     if ang > 180: ang = 360 - ang
#     strng = strng + str(ang) + ","
#     ang = np.rad2deg(2 * acos(rel_8[i, 0]))
#     if ang > 180: ang = 360 - ang
#     strng = strng + str(ang) + ","
#     ang = np.rad2deg(2 * acos(rel_9[i, 0]))
#     if ang > 180: ang = 360 - ang
#     strng = strng + str(ang) + ","
#     strng = strng + obj[i] + "\n"
#     # print(strng)
#     appened_to_file("quat_train.arff", strng)

# data_frame = pd.read_csv("/Users/thisum/Desktop/MagicHand_Data/posture_data/thisum/processed/chamod_test.text", header=None)
#
# back  = normalize(data_frame.loc[:,0:3]  )
# pinky = normalize(data_frame.loc[:,4:7]  )
# ring  = normalize(data_frame.loc[:,8:11] )
# middl = normalize(data_frame.loc[:,12:15])
# index = normalize(data_frame.loc[:,16:19])
# thumb = normalize(data_frame.loc[:,20:23])
# obj = data_frame.loc[:, 24]
#
# rel_p = normalize(qt.q_mult(qt.q_inv(back), pinky))
# rel_r = normalize(qt.q_mult(qt.q_inv(back), ring ))
# rel_m = normalize(qt.q_mult(qt.q_inv(back), middl))
# rel_i = normalize(qt.q_mult(qt.q_inv(back), index))
# rel_t = normalize(qt.q_mult(qt.q_inv(back), thumb))
#
# strng = ""
# for i in range(rel_p.shape[0]):
#     strng = ""
#     ang = np.rad2deg(2*acos(rel_p[i,0]))
#     if ang > 180: ang = 360 - ang
#     strng = str(ang) + ","
#     ang = np.rad2deg(2*acos(rel_r[i,0]))
#     if ang > 180: ang = 360 - ang
#     strng = strng + str(ang) + ","
#     ang = np.rad2deg(2*acos(rel_m[i,0]))
#     if ang > 180: ang = 360 - ang
#     strng = strng + str(ang) + ","
#     ang = np.rad2deg(2*acos(rel_i[i,0]))
#     if ang > 180: ang = 360 - ang
#     strng = strng + str(ang) + ","
#     ang = np.rad2deg(2*acos(rel_t[i,0]))
#     if ang > 180: ang = 360 - ang
#     strng = strng + str(ang) + ","
#     strng = strng + obj[i] + "\n"
#     appened_to_file("quat_test.arff", strng)
#
# # # #
# # # # plt.subplot(1, 3, 1)
# # # # plt.hist(ang1, bins='auto', color='#0504aa',alpha=0.7, rwidth=0.85)
# # # # plt.subplot(1, 3, 2)
# # # # plt.hist(ang2, bins='auto', color='#0504aa',alpha=0.7, rwidth=0.85)
# # # # plt.subplot(1, 3, 3)
# # # # plt.hist(ang3, bins='auto', color='#0504aa',alpha=0.7, rwidth=0.85)
# # # # plt.show()

qt.calc_quat()
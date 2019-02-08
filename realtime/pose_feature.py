import pickle
from math import acos

import numpy as np
import skinematics.quat as qt
from sklearn.preprocessing import normalize

class PostureCalculator:

    def __init__(self):
        self.pose_model = None
        self.postures = ['DB', 'HA', 'CH', 'CK', 'DM', 'RE', 'WR', 'SA', 'SC', 'ST']
        self.means = np.array([141.134222, 149.383951, 141.020560, 107.692086, 120.372743, 127.365823, 155.576642, 157.320636, 157.526890, 53.068179, 61.547594, 58.351241, 28.976731, 42.003817, 24.087737])
        self.stds = np.array([40.484055, 38.493182, 40.123354, 37.112166, 20.858160, 27.590674, 24.564822, 17.358223, 16.093597, 23.840517, 33.530331, 30.036117, 26.778493, 27.418255, 16.038851])
        pkl_filename = "../data/ensemble/models/posture_thisum.pkl"
        with open(pkl_filename,  'rb') as file:
            self.pose_model = pickle.load(file)

    def calculate_all_angles(self, quat_ary):

        back = normalize(np.array(quat_ary[0]).reshape(1, 4))
        pinky = normalize(np.array(quat_ary[1]).reshape(1, 4))
        ring = normalize(np.array(quat_ary[2]).reshape(1, 4))
        middl = normalize(np.array(quat_ary[3]).reshape(1, 4))
        index = normalize(np.array(quat_ary[4]).reshape(1, 4))
        thumb = normalize(np.array(quat_ary[5]).reshape(1, 4))


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

        angles = []
        ang = np.rad2deg(2 * acos(rel_p[0][0]))
        if ang > 180: ang = 360 - ang
        angles.append(ang)

        ang = np.rad2deg(2 * acos(rel_r[0][0]))
        if ang > 180: ang = 360 - ang
        angles.append(ang)

        ang = np.rad2deg(2 * acos(rel_m[0][0]))
        if ang > 180: ang = 360 - ang
        angles.append(ang)

        ang = np.rad2deg(2 * acos(rel_i[0][0]))
        if ang > 180: ang = 360 - ang
        angles.append(ang)

        ang = np.rad2deg(2 * acos(rel_t[0][0]))
        if ang > 180: ang = 360 - ang
        angles.append(ang)

        ang = np.rad2deg(2 * acos(rel_0[0][0]))
        if ang > 180: ang = 360 - ang
        angles.append(ang)

        ang = np.rad2deg(2 * acos(rel_1[0][0]))
        if ang > 180: ang = 360 - ang
        angles.append(ang)

        ang = np.rad2deg(2 * acos(rel_2[0][0]))
        if ang > 180: ang = 360 - ang
        angles.append(ang)

        ang = np.rad2deg(2 * acos(rel_3[0][0]))
        if ang > 180: ang = 360 - ang
        angles.append(ang)

        ang = np.rad2deg(2 * acos(rel_4[0][0]))
        if ang > 180: ang = 360 - ang
        angles.append(ang)

        ang = np.rad2deg(2 * acos(rel_5[0][0]))
        if ang > 180: ang = 360 - ang
        angles.append(ang)

        ang = np.rad2deg(2 * acos(rel_6[0][0]))
        if ang > 180: ang = 360 - ang
        angles.append(ang)

        ang = np.rad2deg(2 * acos(rel_7[0][0]))
        if ang > 180: ang = 360 - ang
        angles.append(ang)

        ang = np.rad2deg(2 * acos(rel_8[0][0]))
        if ang > 180: ang = 360 - ang
        angles.append(ang)

        ang = np.rad2deg(2 * acos(rel_9[0][0]))
        if ang > 180: ang = 360 - ang
        angles.append(ang)

        return angles

    def predict(self, data):
        posture_accuracy = []
        p_x_test = (np.array(data) - self.means) / self.stds + + 0.000001
        prob = self.pose_model.predict_proba(np.array(p_x_test).reshape(1, len(data)))

        for i in range(prob.size):
            posture_accuracy.append((self.postures[i], prob[0][i]))

        max_posture = sorted(posture_accuracy, key=lambda x: x[1], reverse=True)[:2]
        return max_posture

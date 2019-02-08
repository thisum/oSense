import numpy as np
from scipy import fftpack
import scipy.stats as st
import pickle


class ActivityCalculator:

    def __init__(self):
        self.prev_freq = 1.0
        self.average = 0.721270254047
        self.T = 1.0 / 19.0
        self.activities = ['DB', 'HA', 'CH', 'CK', 'DM', 'RE', 'WR', 'SA', 'SC', 'ST']
        self.means = np.array(
            [0.286827, 0.185009, -0.030680, 0.822649, 0.262636, 0.606013, 1.568640, 0.353687, 2.954548, 2.926673,
             4.698834, 4.654227, 5.792701, 24.518072, 24.520013, 5.416271, 5.414804, 3.252592, 369.964503, 0.053878,
             0.053884, 0.011944, 0.011940, 0.007084, 1.148997])
        self.stds = np.array(
            [0.021600, 0.114204, 0.169206, 0.403004, 0.037983, 0.921286, 4.642967, 0.070090, 2.947814, 2.860536,
             3.413851, 3.313654, 3.579248, 34.546295, 34.546537, 10.288605, 10.284373, 7.827789, 113.256324, 0.067207,
             0.067208, 0.020327, 0.020314, 0.015499, 0.872008])

        self.activity_model = None
        pkl_filename = "../data/ensemble/models/activity_thisum.pkl"
        with open(pkl_filename, 'rb') as file:
            self.activity_model = pickle.load(file)

    def calculate_total_accel(self, accel_list):

        total_accel = (sum(map(lambda x: x * x, accel_list))) ** 0.5
        return total_accel - self.average

    def calculate_rms_features(self, s_d):
        mean = np.mean(s_d)
        std = np.std(s_d)
        min = np.min(s_d)
        max = np.max(s_d)
        medi = np.median(s_d)
        skew = st.skew(s_d)
        kurt = st.kurtosis(s_d)
        rms = np.sqrt(np.mean(np.power(s_d, 2)))

        return (mean, std, min, max, medi, skew, kurt, rms)

    def calculate_fft_features(self, f_d):
        flat = f_d.real.flatten()
        flat.sort()

        n = list(f_d.real).index(flat[-1])
        d_f1 = n * (1.0 / (self.T)) / f_d.size
        p_f1 = abs(f_d[n]) ** 2

        n = list(f_d.real).index(flat[-2])
        d_f2 = n * (1.0 / (self.T)) / f_d.size
        p_f2 = abs(f_d[n]) ** 2

        n = list(f_d.real).index(flat[-3])
        d_f3 = n * (1.0 / (self.T)) / f_d.size
        p_f3 = abs(f_d[n]) ** 2

        n = list(f_d.real).index(flat[-4])
        d_f4 = n * (1.0 / (self.T)) / f_d.size
        p_f4 = abs(f_d[n]) ** 2

        n = list(f_d.real).index(flat[-5])
        d_f5 = n * (1.0 / (self.T)) / f_d.size
        p_f5 = abs(f_d[n]) ** 2

        if d_f1 == 0.0:
            d_f1 = d_f2
            d_f2 = d_f3
            d_f3 = d_f4
            d_f4 = d_f5

            p_f1 = p_f2
            p_f2 = p_f3
            p_f3 = p_f4
            p_f4 = p_f5

            n = list(f_d.real).index(flat[-6])
            d_f5 = n * (1.0 / self.T) / f_d.size
            p_f5 = abs(f_d[n]) ** 2

        p_t = np.sum([abs(f_d[x]) ** 2 for x in
                      range(int(np.floor(0.3 * self.T * f_d.size)), int(np.ceil(10 * self.T * f_d.size)))])

        return d_f1, d_f2, d_f3, d_f4, d_f5, p_f1, p_f2, p_f3, p_f4, p_f5, p_t, p_f1 / p_t, p_f2 / p_t, p_f3 / p_t, p_f4 / p_t, p_f5 / p_t

    def calculate_accel_features(self, series):
        hamm = np.hamming(len(series))
        yf = fftpack.fft(series * hamm)

        rms_features = self.calculate_rms_features(series)
        fft_features = self.calculate_fft_features(yf)

        feature_row = [x for x in (rms_features + fft_features)]
        feature_row.append((fft_features[0] / self.prev_freq))

        self.prev_freq = fft_features[0]

        return feature_row

    def predict(self, data):

        activity_accuracy = []
        a_x_test = (np.array(data) - self.means) / self.stds + + 0.000001
        prob = self.activity_model.predict_proba(np.array(a_x_test).reshape(1, len(data)))

        for i in range(prob.size):
            activity_accuracy.append((self.activities[i], prob[0][i]))

        max_activity = sorted(activity_accuracy, key=lambda x: x[1], reverse=True)[:2]
        return max_activity

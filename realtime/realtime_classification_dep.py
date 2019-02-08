import time
import serial

from pose.MagwickCalculator import MagwickCalculator
from realtime.activity_feature import ActivityCalculator
from realtime.pose_feature import PostureCalculator


class RealtimeClassifier:

    def __init__(self, socket_ref):

        self.posture_cal = PostureCalculator()
        self.activity_cal = ActivityCalculator()
        self.cal = [MagwickCalculator(), MagwickCalculator(), MagwickCalculator(), MagwickCalculator(), MagwickCalculator(), MagwickCalculator()]
        self.accel_list = []
        self.dict = {
            'DB': 'Drinking (Bottle)',
            'HA': 'Hammering',
            'CH': 'Chopping',
            'CK': 'Cutting (Knife)',
            'DM': 'Drinking (Mug)',
            'RE': 'Relax',
            'WR': 'Writing',
            'SA': 'Sawing',
            'SC': 'Driving a Screw',
            'ST': 'Stirring'
        }
        self.socket_ref = socket_ref


    def make_ensemble_prediction(self, activity, posture):

        if posture[0][0] in (['DB', 'DM', 'WR', 'SC', 'SA', 'CK', 'ST', 'RE']) and posture[0][1] > 0.950:
            prediction = posture[0][0]

        elif posture[0][1] > activity[0][1] and activity[0][1] - activity[1][1] < 0.9:
            prediction = posture[0][0]
        else:
            prediction = activity[0][0]

        self.socket_ref.emit('newnumber', {'predict': self.dict[prediction], 'posture': str(posture), 'activity': str(activity)}, namespace='/test')


    def onDataAvailable(self, line):

        imu_readings = line.split("#", 6)
        quat = [[],[],[],[],[],[]]
        accel = []

        for imu in imu_readings:
            reading = [float(i) for i in imu.split(',')]
            device = reading[0]

            if device == 1:
                quat[0] = self.cal[0].update(reading[1:])
                accel = reading[1:4]
            elif device == 2:
                quat[1] = self.cal[1].update(reading[1:])
            elif device == 3:
                quat[2] = self.cal[2].update(reading[1:])
            elif device == 4:
                quat[3] = self.cal[3].update(reading[1:])
            elif device == 5:
                quat[4] = self.cal[4].update(reading[1:])
            elif device == 6:
                quat[5] = self.cal[5].update(reading[1:])

        angles = self.posture_cal.calculate_all_angles(quat)
        accel = self.activity_cal.calculate_total_accel(accel)
        self.accel_list.append(accel)
        if len(self.accel_list) > 100:
            self.accel_list.pop(0)
            accel_feature = self.activity_cal.calculate_accel_features(self.accel_list)

            posture_pred = self.posture_cal.predict(angles)
            activity_pred = self.activity_cal.predict(accel_feature)

            self.make_ensemble_prediction(activity_pred, posture_pred)


    def initialize_port_listening(self):

        s = serial.Serial(port='/dev/tty.usbmodem4869681', baudrate=115200)
        calculateQ = False

        while True:
            line = s.readline()
            line = line.decode('utf-8')
            if not calculateQ:
                print(line)
            else:
                self.onDataAvailable(line)

            if not calculateQ and "**##**" in line:
                calculateQ = True
                print('--------------------------------- Setup Done --------------------------------- \n')



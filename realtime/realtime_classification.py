import serial
from aiohttp import web
from threading import *
from socketio import AsyncServer
from pose.MagwickCalculator import MagwickCalculator
from realtime.activity_feature import ActivityCalculator
from realtime.pose_feature import PostureCalculator

posture_cal = PostureCalculator()
activity_cal = ActivityCalculator()
cal = [MagwickCalculator(), MagwickCalculator(), MagwickCalculator(), MagwickCalculator(), MagwickCalculator(), MagwickCalculator()]
accel_list = []
dict = {
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

sio = AsyncServer()
app = web.Application()
sio.attach(app)

@sio.on('message')
async def print_message(sid, message):
    print(message)

async def send_data_to_client(message):
    await sio.emit('message', message)


def make_ensemble_prediction(activity, posture):

    if posture[0][0] in (['DB', 'DM', 'WR', 'SC', 'SA', 'CK', 'ST', 'RE']) and posture[0][1] > 0.990:
        prediction = posture[0][0]

    elif posture[0][1] > activity[0][1] and activity[0][1] - activity[1][1] < 0.9:
        prediction = posture[0][0]
    else:
        prediction = activity[0][0]

    return dict[prediction]


def onDataAvailable(line):

    imu_readings = line.split("#", 6)
    quat = [[],[],[],[],[],[]]
    accel = []

    for imu in imu_readings:
        reading = [float(i) for i in imu.split(',')]
        device = reading[0]

        if device == 1:
            quat[0] = cal[0].update(reading[1:])
            accel = reading[1:4]
        elif device == 2:
            quat[1] = cal[1].update(reading[1:])
        elif device == 3:
            quat[2] = cal[2].update(reading[1:])
        elif device == 4:
            quat[3] = cal[3].update(reading[1:])
        elif device == 5:
            quat[4] = cal[4].update(reading[1:])
        elif device == 6:
            quat[5] = cal[5].update(reading[1:])


    angles = posture_cal.calculate_all_angles(quat)
    accel = activity_cal.calculate_total_accel(accel)
    accel_list.append(accel)
    if len(accel_list) > 100:
        accel_list.pop(0)
        accel_feature = activity_cal.calculate_accel_features(accel_list)

        posture_pred = posture_cal.predict(angles)
        activity_pred = activity_cal.predict(accel_feature)

        pred = make_ensemble_prediction(activity_pred, posture_pred)
        send_data_to_client(pred + " pos: " + str(posture_pred) + " activ: " + str(activity_pred))


def start_classifier():
    s = serial.Serial(port='/dev/tty.usbmodem4869681', baudrate=115200)
    calculateQ = False
    count = 0

    while True:
        line = s.readline()
        line = line.decode('utf-8')
        if not calculateQ:
            print(line)
        else:
            onDataAvailable(line)

        if not calculateQ and "**##**" in line:
            calculateQ = True
            print('--------------------------------- Setup Done --------------------------------- \n')



async def index(request):
    with open('index.html') as f:
        return web.Response(text=f.read(), content_type='text/html')

def start_webapp():
    app.router.add_get('/', index)
    web.run_app(app)


class classifierThread(Thread):
    # initialize class

    def run(self):
        start_classifier()

thread1 = classifierThread()
thread1.start()

start_webapp()

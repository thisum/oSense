import time

import serial

from pose.MagwickCalculator import MagwickCalculator

s = serial.Serial(port='/dev/tty.usbmodem4869681', baudrate=115200)
count = 50000
lastUpdate = 1
printLine = True
calculateQ = False

ary = [[0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0]]

cal1 = MagwickCalculator()
cal2 = MagwickCalculator()

def calculateRelativeAngles( fAry, bAry):
    print(format((fAry[0] - bAry[0]),'.2f') + "," + format((fAry[1] - bAry[1]),'.2f') + "," + format((fAry[2] - bAry[2]),'.2f') )

while count>0:
    line = s.readline()
    if printLine:
        print(line)

    if not calculateQ and "**##**" in str(line):
        calculateQ = True
        printLine = False

    elif calculateQ:
        line = line.decode().rstrip()
        line = line.split("#")
        # print(line[1])
        # print("yaw, pitch, roll: " + line[1])
        line = line[0]

        lst_int = [float(x) for x in line.split(",")]

        if lst_int[0] == 1:
            st = int(round(time.time() * 1000))
            cal1.calculateDeltat(st)
            cal1.update((lst_int[1],lst_int[2],lst_int[3]),(lst_int[4],lst_int[5],lst_int[6]),(lst_int[7], lst_int[8], lst_int[9]))
            ary[0] = cal1.getAngles()
            print("1: " + str(ary[0]))

        elif lst_int[0] == 2:
            st = int(round(time.time() * 1000))
            cal2.calculateDeltat(st)
            cal2.update((lst_int[1],lst_int[2],lst_int[3]),(lst_int[4],lst_int[5],lst_int[6]),(lst_int[7], lst_int[8], lst_int[9]))
            ary[1] = cal2.getAngles()
            print("2: " + str(ary[1]))

            calculateRelativeAngles(ary[0], ary[1])





    # line = line.decode().rstrip()
    # line = line.split("#")
    # cal = line[1].split("**")
    # if len(cal) > 1:
    #     print(cal[1])
    # # print(line[1])
    # # print("yaw, pitch, roll: " + line[1])
    # line = line[0]
    # # print(line)
    #
    #
    # lst_int = [float(x) for x in line.split(",")]
    #
    # if lst_int[0] == 1:
    #     st = int(round(time.time() * 1000))
    #     deltat = (st - lastUpdate)/1000
    #     lastUpdate = st
    #     cal1.update((lst_int[1],lst_int[2],lst_int[3]),(lst_int[4],lst_int[5],lst_int[6]),(lst_int[7], lst_int[8], lst_int[9]), deltat)
    #
    # print("----------------------")
    count = count - 1

#

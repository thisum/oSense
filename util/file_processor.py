import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy import fftpack
import os

def appened_to_file(flie_name, line):
    with open(flie_name, "a") as write_file:
        write_file.write(line)

def prepare_data():

    for root, dirs, files in os.walk("/Users/thisum/Desktop/MagicHand_Data/sequence_data/data/thisum/"):
        for i, filename in enumerate(files):

            if filename.startswith("."):
                continue

            file = open(root + "/" + filename, "r")
            lines = file.readlines()
            for line in lines:

                if 'spoon stirring' in line:
                    line = line.replace('spoon stirring', 'spoon_stirring')
                elif 'bottle drinking' in line:
                    line = line.replace('bottle drinking', 'bottle_drinking')
                elif 'mug drinking' in line:
                    line = line.replace('mug drinking', 'mug_drinking')
                elif 'knife chopping' in line:
                    line = line.replace('knife chopping', 'knife_chopping')
                elif 'nailing hammer' in line:
                    line = line.replace('nailing hammer', 'hammer_hammering')
                elif 'saw cutting' in line:
                    line = line.replace('saw cutting', 'saw_sawing')
                elif 'writing pen' in line:
                    line = line.replace('writing pen', 'pen_writing')
                elif 'screwdriver-without-release' in line:
                    line = line.replace('screwdriver-without-release', 'screwdriver_screwing(no-release)')
                elif 'screwdriver-release' in line:
                    line = line.replace('screwdriver-release', 'screwdriver_screwing(release)')

                appened_to_file('/Users/thisum/Desktop/MagicHand_Data/sequence_data/data/thisum/mod_' + filename, line)


prepare_data()
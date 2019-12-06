import math
import numpy as np
import matplotlib.pyplot as plt
import csv
import errno
import os
import pathlib
import json

def ensure_directory_exists(base_directory):
    """
    Makes a directory if it does not exist
    """
    try:
        os.makedirs(base_directory)
    except OSError as ex:
        if ex.errno != errno.EEXIST:
            raise ex

class Time_Series_Statistics_Store(object):
    """
    Logs time series data.
    Header should represent every column in data.
    For example:
        epoch | loss
        0     | 1
        1     | 0.5
        2     | 0.3
    """
    def __init__(self, header):
        self.statistics = []
        self.header = header
    def add_statistics(self, data):
        if len(data) != len(self.header):
            raise ValueError("Data length does not match header")
        self.statistics.append(data)
    def dump(self, dump_filename="statistics.csv"):
        p = pathlib.Path(dump_filename)
        if len(p.parts) > 1:
            dump_dirs = pathlib.Path(*p.parts[:-1])
            ensure_directory_exists(str(dump_dirs))
        with open(dump_filename, "w") as csvfile:
            wr = csv.writer(csvfile)
            wr.writerow(self.header)
            for stat in self.statistics:
                wr.writerow(stat)
    def summarize_last(self):
        summary = ""
        for i in range(len(self.header)):
            if isinstance(self.statistics[-1][i], float):
                summary += "{}: {:.3f},".format(self.header[i], self.statistics[-1][i])
            else:
                summary += "{}: {},".format(self.header[i], self.statistics[-1][i])
        return summary


def save_dict(dict, file_name):
    w = csv.writer(open(file_name, "w"))
    for key, val in dict.items():
        w.writerow([key, val])
    # w.close()


def distance(pos1, pos2):
    delta_pos = pos1 - pos2
    return np.sqrt(np.sum(np.square(delta_pos)))


def plot(simple_model, DQL, ddpg):
    plt.plot(simple_model, 'g', label='simple model')
    plt.plot(DQL, 'r', label='deep Q learning')
    plt.plot(ddpg, 'b', label='DDPG')
    plt.legend(loc='upper right')
    plt.xlabel('time')
    plt.ylabel('mean_distance')
    plt.show()


if __name__ == '__main__':
    speed = np.array((0,0))
    speed = speed / np.sqrt(np.square(speed[0]) + np.square(speed[1]))
    print(speed)
    sm = np.ones((100,))


    def func_1(i):
        return 100. / (i + 1) + 1


    def func_2(i):
        return 100. / (i / 2. + 1) + 3


    lin = np.linspace(0, 100, 100)
    dql = func_2(lin)
    ddpg = func_1(lin)
    plot(sm, dql, ddpg)

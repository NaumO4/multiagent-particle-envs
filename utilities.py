import math
import numpy as np
import matplotlib.pyplot as plt


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
    sm = np.ones((100,))


    def func_1(i):
        return 100. / (i + 1) + 1


    def func_2(i):
        return 100. / (i / 2. + 1) + 3


    lin = np.linspace(0, 100, 100)
    dql = func_2(lin)
    ddpg = func_1(lin)
    plot(sm, dql, ddpg)

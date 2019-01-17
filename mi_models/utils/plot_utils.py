# -*- coding: utf-8 -*-
# @time    : 2018/9/10 17:15
# @author  : huangyu10@cmschina.com.cn
# @file    : plot_utils.py

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_2D(values=[], style='plot', x_legend='', y_legend='', x_label='', y_label='', saved_path=None,
            title=''):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    for value in values:
        x_values, y_values = value
        if style == 'plot':
            plt.scatter(x_values, y_values, label=u'', linestyle='-')
        elif style == 'bar':
            plt.bar(x_values, y_values, color='b', alpha=0.65, label=u'')
    plt.gcf().autofmt_xdate()
    plt.legend([x_legend, y_legend])
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if saved_path:
        plt.savefig(saved_path)
    else:
        plt.show()


def plot_3D(values=[], legends=[], labels=[], saved_path=None, title=''):
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection='3d')
    for idx, value in enumerate(values):
        x_values, y_values, z_values = value
        ax.scatter(x_values, y_values, z_values, label=legends[idx])
    plt.title(title)
    plt.legend(loc='upper right')
    if saved_path:
        plt.savefig(saved_path)
    else:
        plt.show()

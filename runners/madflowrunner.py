import runners.data_loader as data_loader
from runners.mad import *
import numpy as np
from pandas import DataFrame
import matplotlib
import time
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.dates as md
import datetime as dt

def data_it(raw_data, block_size=5):
    for i in range(len(raw_data) - block_size):
        eval_data = raw_data[i: i + block_size]
        yield eval_data


def run_mad_anom_detection(v, training_data_len=16):
    df = get_confidence_band(v[:training_data_len], v[training_data_len:], np.mean, 0.997)

    return {
            "lower": df['lower'],
            "upper": df['upper']
            }


def run_mad_anom_detection_no_time(v, training_data_len=120):
    df = get_confidence_band(v[:training_data_len], v[training_data_len:], 0.997)

    return {
            "lower": df['lower'],
            "upper": df['upper']
            }


if __name__ == "__main__":
    data = data_loader.load_data('../data/demo7.csv')
    training_len = 60
    r1 = run_mad_anom_detection(np.asarray(data.values).tolist(), training_len)
    timestamps = np.asarray(data.index).tolist()
    values = np.asarray(data.values).tolist()
    dates = [dt.datetime.fromtimestamp(ts/1000000000) for ts in timestamps]
    plt.figure(figsize=[20, 8])
    plt.subplots_adjust(bottom=0.2)
    plt.xticks(rotation=25)
    ax = plt.gca()
    xfmt = md.DateFormatter('%Y-%m-%d')
    ax.xaxis.set_major_formatter(xfmt)

    for i in range(training_len, len(values)):
        if r1['upper'][i-training_len] < values[i] or r1['lower'][i-training_len] > values[i]:
            plt.plot(dates[i], values[i], 'ro')
    plt.fill_between(dates[training_len:], r1['upper'], r1['lower'], color='lightgray')
    plt.plot(dates, values)

    plt.show()

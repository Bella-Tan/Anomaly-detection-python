import runners.data_loader as data_loader
from runners.mad import (get_confidence_band, get_mad_outlier)
from timer import Timer
from pandas import DataFrame
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def data_it(raw_data, block_size = 5):
    for i in range(len(raw_data) - block_size):
        eval_data = raw_data[i : i + block_size]
        yield eval_data


def run_mad_anom_detection_flow(file_path, block_size = 5):
    sw = Timer()
    sw.start()
    data = data_loader.load_data(file_path)
    data_flow = data_it(data, block_size)

    df = []
    try:
        for data_block in data_flow:
            outerliner = get_mad_outlier(data_block)
            df.append(outerliner)
            
    except StopIteration as ex:
        print(ex)
    print(sw.stop("using time: "))

    columns = ['max', 'min', 'raw', 'ts', 'med']
    df = DataFrame(df, columns=columns)
    plt.fill_between(df['ts'].values, df['max'].values, df['min'].values, color='lightgray')
    plt.plot(df['ts'].values, df['raw'].values)
    plt.plot(df['ts'].values, df['med'].values)

    for i in range(len(df)):
        raw_row = df.values[i]
        if raw_row[2] < raw_row[1] or raw_row[2] > raw_row[0]:
            plt.plot(raw_row[3], raw_row[2], 'ro')

    plt.show()


def run_mad_anom_detection_after_m4(timestamps, values):
    df = get_confidence_band(values[:5], values[5:])

    return {'timestamps': timestamps[5:],
            'values': values[5:],
            "lower": df['lower'],
            "upper": df['upper']
            }


if __name__ == "__main__":
    sw = Timer()
    sw.start()
    data = data_loader.load_data_not_parse_time('../data/santaba-demo2.csv')
    result = run_mad_anom_detection_after_m4(np.asarray(data.index).tolist(), np.asarray(data.values).tolist())
    print(sw.stop("using time: "))
    plt.fill_between(result["timestamps"], result['upper'], result['lower'], color='lightgray')
    plt.plot(result['timestamps'], result['values'])

    for i in range(len(result['values']) - 5):
        raw_row = result['values'][5:]
        if raw_row[2] < raw_row[1] or raw_row[2] > raw_row[0]:
            plt.plot(result['timestamps'], result['values'], 'ro')

    plt.show()

import runners.data_loader as data_loader
from runners.mad import *
from lib.timer import Timer
from pandas import DataFrame
import numpy as np

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def data_it(raw_data, block_size=5):
    for i in range(len(raw_data) - block_size):
        eval_data = raw_data[i: i + block_size]
        yield eval_data


def run_mad_anom_detection(v, training_data_len=720):
    df = get_confidence_band(v[:training_data_len], v[training_data_len:], np.mean, 0.99)

    return {
            "lower": df['lower'],
            "upper": df['upper']
            }


def run_mad_anom_detection_no_time(v, training_data_len=120):
    df = get_confidence_band(v[:training_data_len], v[training_data_len:], 0.99)

    return {
            "lower": df['lower'],
            "upper": df['upper']
            }


def run_mad_with_history(timestamps, v, len_train):
    lower = []
    upper = []
    data = [{"uuid": 12345, "values":v[:len_train], "windowSize":len_train}]
    r = get_confidence_band_with_historic(data, 0.99)
    data[0] = r[data[0]["uuid"]]
    data[0]["uuid"] = 12345
    for i in range(len_train, len(v)-1):

        data[0]["values"] = v[i]
        r = get_confidence_band_with_historic(data, 0.99)
        lower.append(r[0])
        upper.append(r[1])

    return {'timestamps': timestamps[len_train:],
            "lower": lower,
            "upper": upper
            }




if __name__ == "__main__":
    data = data_loader.load_data('../data/linuxping.csv')
    count = 0
    training_len = 120
    r1 = run_mad_anom_detection(np.asarray(data.values).tolist(), training_len)
    timestamps = np.asarray(data.index).tolist()[training_len:]
    values = np.asarray(data.values).tolist()[training_len:]
    anomalies1 = []
    anomalies2 = []
    mean_div = []
    mad_div = []
    for i in range(len(values)):
        if r1['upper'][i] < values[i] or r1['lower'][i] > values[i]:
            anomalies1.append(i)

    print(mean_div)
    print(f"anomaly points count by mad with real data : {len(anomalies1)}")
    result = []
    for i in anomalies1:
        if i in anomalies2:
            result.append(i)

    print(f"common anomaly points count in both results : {len(result)}")
    # fig, axs = plt.subplots(1, 1)
    # fig.set_size_inches((16, 8))
    plt.fill_between(timestamps, r1['upper'], r1['lower'], color='lightgray')
    plt.plot(timestamps, values)

    plt.show()

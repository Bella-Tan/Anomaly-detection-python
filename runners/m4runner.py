import runners.data_loader as data_loader
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
SAMPLE_FILE_PATH = '../data/santaba-demo5.csv'
import numpy as np
from runners.madflowrunner import run_mad_anom_detection_no_time,run_mad_anom_detection


def reduce(values):
    result = []
    maximum = max(values)
    minimum = min(values)
    end = -1
    start = -1
    for i in range(len(values)):
        if values[i] is not None:
            end = i
            if start == -1:
                start = i
                result.append(i)
                continue
            if maximum == values[i] or minimum == values[i]:
                result.append(i)
        else:
            continue
    result.append(end)
    return np.unique(result).tolist()


def aggregation(timestamps, values, window_size=8):
    values1 = []
    values2 = []
    if type(values) is not list or type(timestamps) is not list or len(timestamps) != len(values):
        raise ValueError("Bad data format for training.")
    for i in range(int(len(values) / window_size)):
        index = reduce(values[i*window_size:i*window_size+window_size])
        for j in range(len(index)):
            values1.append(timestamps[i * window_size + index[j]])
            values2.append(values[i * window_size + index[j]])
    return values1, values2


def aggregation1( values, window_size=120):
    values2 = []
    if type(values) is not list:
        raise ValueError("Bad data format for training.")
    for i in range(int(len(values) / window_size)):
        index = reduce(values[i*window_size:i*window_size+window_size])
        for j in range(len(index)):
            values2.append(values[i * window_size + index[j]])
    return values2


def run_mad_with_m4(v, training_data_len=720, window_size=120):
    lowers = []
    uppers = []
    training_data = aggregation1(v[:training_data_len], window_size)
    test_data = v[training_data_len:]
    j = 0
    for i in range(len(test_data)):
        _data = []
        if i % 120 == 0 and i>0:
            j = i
            training_data = training_data[4:]
        _data = training_data
        aggragated = aggregation1(test_data[j:i], window_size)
        for x in aggragated:
            training_data.append(x)
            _data.append(x)
        _data.append(v[i])

        r2 = run_mad_anom_detection_no_time(_data, len(_data)-1)
        lowers.append(r2["lower"][0])
        uppers.append(r2["upper"][0])
    return {
            "lower": lowers,
            "upper": uppers
        }


if __name__ == '__main__':
    data = data_loader.load_data_not_parse_time(SAMPLE_FILE_PATH)
    length = 480
    timestamps = np.asarray(data.index).tolist()
    values = np.asarray(data.values).tolist()
    r1 = run_mad_anom_detection(values, length)
    r2 = run_mad_with_m4(values, length, 120)
    fig, axs = plt.subplots(3, 1)
    fig.set_size_inches((16, 8))

    axs[0].fill_between(timestamps[length:], r1['upper'], r1['lower'], color='lightgray')
    axs[0].plot(timestamps[length:], values[length:])
    axs[1].fill_between(timestamps[length:], r2['upper'], r2['lower'], color='lightblue')
    axs[1].plot(timestamps[length:], values[length:])
    axs[2].plot(timestamps, values)
    plt.show()


# if __name__ == '__main__':
#     data = data_loader.load_data_not_parse_time(SAMPLE_FILE_PATH)
#     (timestamps, values) = aggregation(np.asarray(data.index).tolist(), np.asarray(data.values).tolist(), 120)
#     fig, axs = plt.subplots(2, 1)
#     axs[0].plot(data)
#     print(f'raw data length: {len(data.values)}')
#     axs[1].plot(values)
#     print(f'aggregated data length: {len(values)}')
#     plt.show()
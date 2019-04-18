import runners.data_loader as data_loader
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
SAMPLE_FILE_PATH = '../data/santaba-demo5.csv'
import numpy as np


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


if __name__ == '__main__':
    data = data_loader.load_data_not_parse_time(SAMPLE_FILE_PATH)
    (timestamps, values) = aggregation(np.asarray(data.index).tolist(), np.asarray(data.values).tolist(), 8)
    fig, axs = plt.subplots(2, 1)
    axs[0].plot(data)
    print(f'raw data length: {len(data.values)}')
    axs[1].plot(values)
    print(f'aggregated data length: {len(values)}')
    plt.show()


import pandas as pd
import numpy as np

def dparserfunc(date):
    return pd.datetime.strptime(date, '%Y/%m/%d %H:%M:%S')


def load_data(path):
    data = pd.read_csv(path, index_col='timestamps', parse_dates=True, sep=',', usecols=['timestamps', 'values'], squeeze=True, date_parser=dparserfunc)
    return data


def load_data_not_parse_time(path):
    data = pd.read_csv(path, index_col='timestamps', parse_dates=False, sep=',', usecols=['timestamps', 'values'],
                   squeeze=True)
    return data


def load_data_as_list(path):
    data = load_data(path)
    result = {}
    result.timestamps = np.asarray(data.index).tolist()
    result.values = np.asarray(data.values).tolist()
    return result


def data_it(raw_data, block_size = 5):
    for i in range(len(raw_data) - block_size):
        eval_data = raw_data[i : i + block_size]
        yield eval_data

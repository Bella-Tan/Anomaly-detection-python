import runners.data_loader as data_loader
import pandas as pd
import matplotlib
import numpy as np
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

SAMPLE_FILE_PATH = '../data/demo8-missing-data.csv'
pd.options.mode.use_inf_as_na = True


def interpolate(values):
    df = pd.DataFrame({'values': values})
    print(df)
    data = df.astype(float).interpolate(method='linear', limit_direction='both', axis=0)
    print(data)
    return np.asarray(data['values']).tolist()


if __name__ == '__main__':
    (timestamps, values) = data_loader.load_data_as_list(SAMPLE_FILE_PATH)
    fig, axs = plt.subplots(2, 1, sharex=True)
    axs[0].plot(timestamps, values)
    values = interpolate(values)
    axs[1].plot(timestamps, values)
    plt.show()

import math
import runners.data_loader as data_loader
import pandas as pd
import matplotlib.pyplot as plt

figure_size = (12, 4)
seasonal_period = 131
SAMPLE_FILE_PATH = '../data/demo1.csv'


if __name__ == '__main__':
    data = data_loader.load_data(SAMPLE_FILE_PATH).rolling(1).mean()

    fig, axs = plt.subplots(4, 1)
    axs[0].plot(data)
    raw_data_length = len(data)
    season_count = int(raw_data_length / seasonal_period)
    print(f'raw data length: {raw_data_length}, period: {seasonal_period}, count: {season_count}')
    season_values = []
    trend_values = data.rolling(seasonal_period).mean()
    axs[1].plot(trend_values)
    for i in range(seasonal_period):
        current_values = []
        current_index = i
        while current_index < raw_data_length:
            current_v = data.values[current_index]
            current_values.append(current_v)
            current_index += seasonal_period
        season_values.append(pd.Series(current_values).mean())

    season_values = season_values * season_count
    axs[2].plot(season_values)
    min_count = min(len(data), len(season_values), len(trend_values))
    loess_values = data.values[:min_count] - season_values[:min_count] - trend_values[:min_count]
    axs[3].plot(loess_values)

    fig.tight_layout()
    plt.show()

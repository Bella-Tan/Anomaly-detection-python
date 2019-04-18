import numpy as np
from timer import Timer
import runners.data_loader as data_loader
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
SAMPLE_FILE_PATH = '../data/santaba-demo5.csv'


def arima(timestamps, values):
    size = int(len(values) * 0.66)
    train, test = values[0:size], values[size:len(values)]
    history = [x for x in train]

    predictions = []
    for t in range(len(test)):
        model = ARIMA(history, order=(5, 1, 0))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat[0])
        obs = test[t]
        history.append(obs)
    error = mean_squared_error(test, predictions)
    print('Test MSE: %.3f' % error)
    return timestamps[size:len(values)], predictions, test


if __name__ == '__main__':
    sw = Timer()
    sw.start()
    data = data_loader.load_data(SAMPLE_FILE_PATH)
    (timestamps, values) = np.asarray(data.index).tolist(), np.asarray(data.values).tolist()
    (timestamps, predicted, expected) = arima(timestamps, values)
    print(sw.elapsed(f'data length: {len(values)},  using time:'))
    plt.plot(timestamps, predicted)
    plt.plot(timestamps, expected)
    plt.fill_between(timestamps, predicted, expected, color='lightgray')
    plt.show()

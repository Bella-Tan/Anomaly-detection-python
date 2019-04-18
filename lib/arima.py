from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error


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



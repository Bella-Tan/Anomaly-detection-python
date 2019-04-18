import runners.data_loader as data_loader
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

figure_size = (12, 4)
seasonal_period = 60
SAMPLE_FILE_PATH = '../data/santaba-demo5.csv'


if __name__ == '__main__':
    data = data_loader.load_data(SAMPLE_FILE_PATH)
    result = seasonal_decompose(data, model='multiplicative', freq=10)
    result.plot()
    plt.show()

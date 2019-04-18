import pandas as pd
import breakout_detection
import runners.data_loader as data_loader
import matplotlib.pyplot as plt

SAMPLE_FILE_PATH = '../data/santaba-demo5.csv'


if __name__ == '__main__':
    data = data_loader.load_data(SAMPLE_FILE_PATH)
    plt.style.use('fivethirtyeight')
    edm_multi = breakout_detection.EdmMulti()
    max_snp = max(max(data.values), 1)
    Z = [x/float(max_snp) for x in data.values]
    edm_multi.evaluate(Z, min_size=64, beta=0.0001, degree=1)
    plt.plot(edm_multi)
    for i in edm_multi.getLoc():
        plt.axvline(data.values.index[i], color='#FF4E24')

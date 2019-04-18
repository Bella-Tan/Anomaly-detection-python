import numpy as np
from lib.timer import Timer
import breakout_detection
import runners.data_loader as data_loader
import matplotlib.pyplot as plt

SAMPLE_FILE_PATH = '../data/demo7.csv'


if __name__ == '__main__':
    sw = Timer()
    sw.start()
    data = data_loader.load_data(SAMPLE_FILE_PATH)
    edm_multi = breakout_detection.EdmMulti()
    max_snp = max(max(data.values), 1)
    # Z = [x/float(max_snp) for x in data.values]
    Z = [x for x in data.values]
    edm_multi.evaluate(Z, min_size=24, beta=0.001, degree=1)
    print(sw.elapsed(f'data length: {len(data.values)},  using time:'))
    plt.plot(np.asarray(data.index).tolist(),Z)
    result = edm_multi.getLoc()
    print(result)
    for i in result:
        plt.axvline(np.asarray(data.index).tolist()[i], color='#FF4E24')
        # plt.plot(np.asarray(data.index).tolist()[i], np.asarray(data.values).tolist()[i], 'ro')
    plt.show()

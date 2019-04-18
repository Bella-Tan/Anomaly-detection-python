
from runners.madflowrunner import run_mad_anom_detection_after_m4
from timer import Timer
from runners.m4runner import run_m4_aggregation
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
SAMPLE_FILE_PATH = './data/santaba-demo2.csv'


def main():
    sw = Timer()
    sw.start()
    result1 = run_m4_aggregation(SAMPLE_FILE_PATH)
    result = run_mad_anom_detection_after_m4(result1[0], result1[1])
    print(sw.stop("using time: "))
    plt.fill_between(result["timestamps"], result['upper'], result['lower'], color='lightgray')
    plt.plot(result['timestamps'], result['values'])

    for i in range(len(result['values']) - 5):
        raw_row = result['values'][5:]
        if raw_row[2] < raw_row[1] or raw_row[2] > raw_row[0]:
            plt.plot(result['timestamps'], result['values'], 'ro')

    plt.show()


if __name__ == '__main__':
    main()


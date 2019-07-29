import numpy as np
from scipy.stats import norm

def __apply_along_mean_or_median(function, values):
    try:
        return np.apply_along_axis(np.mean, 0, values)
    except BaseException as e:
        raise e


def __calculate(values, function=np.mean, threshold=0.99):
    try:
        values = [x for x in values if x is not None]
        if type(values) is not list:
            raise ValueError("Bad data format for training.")
        if len(values) < 1:
            return "NaN", "NaN", None
        median = __apply_along_mean_or_median(function, values)
        deviation = list(np.absolute(values - median))
        mad = __apply_along_mean_or_median(function, deviation)
        thresh_mad = norm.ppf(threshold) * mad / threshold
        lower = median - thresh_mad
        upper = median + thresh_mad
        return lower, upper, np.array(median).tolist() * 1.0
    except Exception as e:
        raise e


def get_confidence_band(training_values, testing_values, function=np.mean, threshold=0.99):
    try:
        values = np.concatenate((training_values, testing_values), axis=None).tolist()
        lowers = []
        uppers = []
        temp = [x for x in training_values if x is not None]
        if len(temp) < 2:
            raise ValueError("Training dataSet has less than two no-NaN data.")
        median = __apply_along_mean_or_median(function, temp)
        median = np.array(median).tolist() * 1.0
        for i in range(len(testing_values)):
            if testing_values[i] is None:
                testing_values[i] = median
            result = __calculate(values[i:i + len(training_values)], function, threshold)
            lowers.append(result[0])
            uppers.append(result[1])
            median = result[2] if result[2] is not None else median
            if result[0] != "NaN" and testing_values[i] < result[0] or \
                    result[1] != "NaN" and testing_values[i] > result[1]:
                testing_values[i] = median
        return {
            "lower": lowers,
            "upper": uppers
        }

    except BaseException as e:
        raise e


def __calculate_mean_and_mad(values, threshold=0.99):
    try:
        values = [x for x in values if x is not None]
        if type(values) is not list:
            raise ValueError("Bad data format for training.")
        if len(values) < 1:
            return "NaN", "NaN"
        mean = np.mean(values)
        deviation = list(np.absolute(values - mean))
        mad = np.mean(deviation)
        thresh_mad = norm.ppf(threshold) * mad / threshold
        return np.array(mean).tolist() * 1.0, mad, thresh_mad
    except Exception as e:
        raise e


def get_confidence_band_with_historic(data_array, threshold=0.99):
    try:
        if type(data_array) is not list or data_array is None:
            raise ValueError("bad request data.")
        out = {}
        for data in data_array:
            if type(data["values"]) is list and len(data["values"]) > 1:
                sum_mean = sum(data["values"])
                out[data["uuid"]] = {
                    "lower": "NaN",
                    "upper": "NaN",
                    "sumMean": sum_mean,
                    "sumMad": 0,
                    "sumMeanCache": sum_mean,
                    "windowSize": data["windowSize"],
                    "n": len(data["values"])
                }
            elif data["n"] is not None and data["sumMean"] is not None and data["sumMad"] is not None:
                n = data["n"] + 1
                sum_mean = data["sumMean"] + data["values"]
                mean = sum_mean / n
                sum_mad = data["sumMad"] + np.absolute(data["values"] - mean)
                mad = sum_mad / (n - data["windowSize"])
                thresh_mad = norm.ppf(threshold) * mad / threshold
                sum_mean_cache = data["sumMeanCache"]
                if n == data["windowSize"] * 2:
                    n = data["windowSize"]
                    sum_mean = data["sumMean"] - data["sumCache"]
                    sum_mean_cache = sum_mean
                    data["sumMad"] = 0
                out[data["uuid"]] = {
                    "lower": mean - thresh_mad,
                    "upper": mean + thresh_mad,
                    "sumMean": sum_mean,
                    "sumMad": sum_mad,
                    "sumMeanCache": sum_mean_cache,
                    "windowSize": data["windowSize"],
                    "n": n
                }
            else:
                raise ValueError("bad request data.")
        return out
    except BaseException as e:
        raise e


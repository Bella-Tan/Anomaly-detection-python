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
        print( median, mad, thresh_mad)
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
        median = __apply_along_mean_or_median(np.mean, temp)
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
                values[len(training_values) + i] = median
        return {
            "lower": lowers,
            "upper": uppers
        }

    except BaseException as e:
        raise e
#
#
# def __calculate_mean_and_mad(values, threshold=0.99):
#     try:
#         values = [x for x in values if x is not None]
#         if type(values) is not list:
#             raise ValueError("Bad data format for training.")
#         if len(values) < 1:
#             return "NaN", "NaN"
#         mean = np.median(values)
#         deviation = list(np.absolute(values - mean))
#         mad = np.median(deviation)
#         thresh_mad = norm.ppf(threshold) * mad / threshold
#         return np.array(mean).tolist() * 1.0, mad, thresh_mad
#     except Exception as e:
#         raise e

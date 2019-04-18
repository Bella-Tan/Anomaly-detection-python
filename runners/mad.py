import numpy as np
from scipy.stats import norm


def mad_based_outlier(points, thresh=3.5):
    if type(points) is list:
        points = np.asarray(points)
    if len(points.shape) == 1:
        points = points[:, None]
    med = np.median(points, axis=0)
    abs_dev = np.absolute(points - med)

    med_abs_dev = np.median(abs_dev)

    mod_z_score = norm.ppf(0.75) * abs_dev / med_abs_dev
    return mod_z_score > thresh


def get_mad_outlier(points, thresh=3.5):
    last_index = len(points) - 1

    last_ts = points.index[last_index]
    if type(points) is list:
        points = np.asarray(points)
    if len(points.shape) == 1:
        points = points[:, None]
    med = np.median(points, axis=0)
    abs_dev = np.absolute(points - med)

    med_abs_dev = np.median(abs_dev)

    abs_dev_bounds = thresh * med_abs_dev / norm.ppf(0.75)
    abs_upper = med + abs_dev_bounds
    abs_lower = med - abs_dev_bounds

    abs_upper = abs_upper[0] if abs_upper.size == 1 else None
    abs_lower = abs_lower[0] if abs_lower.size == 1 else None

    return abs_upper, abs_lower, points[last_index][0], last_ts, med


def __apply_along_mean_or_median(function, values):
    try:
        return np.apply_along_axis(np.mean, 0, values)
    except BaseException as e:
        raise e


def __calculate(values, function, threshold=0.95):
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

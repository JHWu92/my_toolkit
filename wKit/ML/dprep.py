# coding=utf-8
import numpy as np


def discretize(measurement, how='std', alpha=(0, 0.5, 1, 2), nbins=10, retn_bins=False):
    """
    Parameters
    ----------
    measurement: array-like of discrete or continuous measurement
    how: {'std', 'bin'}, default std
    alpha: array-like, non-nagtive only, default (0, 0.5, 1, 2)
        used when how='std', discretize data by standard deviation: mean+/-alpha*std (alpha=1 or 0 or 2 or 0.5)
    nbins: int, default 10.
        used when how='bin', discretize data by equidistance bins in [min, max].
    retn_bins: boolean, default False
        if True, return (discreted_measurement, bins)

    Return
    ----------
    discrete_measurement, or (discrete_measurement, bins) if retn_bins

    """

    if how == 'std':
        mean, std = np.mean(measurement), np.std(measurement)
        alpha = np.array(alpha)
        alpha = np.concatenate([-alpha[alpha != 0][::-1], alpha])
        bins = [(mean + a * std) for a in alpha]
    elif how == 'bin':
        min_, max_ = np.min(measurement), np.max(measurement)
        bins = np.linspace(min_, max_, nbins)
    else:
        raise ValueError("allowed how: {'std', 'bin'}")

    dmeasure = np.digitize(measurement, bins)

    if retn_bins:
        return dmeasure, bins
    else:
        return  dmeasure
# coding=utf-8
import numpy as np
import pandas as pd

from wKit.stat.infer import stat_dtype


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
        return dmeasure


def discretize_features(arr2d, how='std', alpha=(0, 0.5, 1, 2), nbins=10):
    """
    discretize features, by arr2d's columns.
    If a feature is inferred to possibly be nominal, it won't go through discretize()

    Parameters
    ----------
    arr2d: 2d-array-like, (n_samples, n_features)
    how: {'std', 'bin'}, default std
    alpha: array-like, non-nagtive only, default (0, 0.5, 1, 2)
        used when how='std', discretize data by standard deviation: mean+/-alpha*std (alpha=1 or 0 or 2 or 0.5)
    nbins: int, default 10.
        used when how='bin', discretize data by equidistance bins in [min, max].
    """

    # TODO: add parameter: dtypes, list or dict. User can specify dtypes instead of inferring.

    idx = None
    if isinstance(arr2d, pd.DataFrame):
        idx = arr2d.index
        col = arr2d.columns
        arr2d = arr2d.values

    _, n_features = arr2d.shape

    discrete = []
    for i in range(n_features):
        arr = arr2d[:, i].copy()
        infer, _ = stat_dtype(arr)
        if 'nominal' not in infer:
            arr = [float(x) for x in arr]
            discrete.append(discretize(arr, how=how, alpha=alpha, nbins=nbins))
        else:
            discrete.append(arr)

    darr2d = np.array(discrete).T

    if idx is not None:
        darr2d = pd.DataFrame(darr2d, index=idx, columns=col)
    return darr2d


def fillna_group_mean(df, group):
    """
    df: feature dataset
    group: columns = ['index', 'name']
    index of group == index of df

    Examples
    ---------
    >>> data = [
        (1, 2, 3, 4, 5, 6, 7, 8),
        (38, 18, None, 193, 98, 874, None, None),
        (939,840,19, 8028, 9280, None, 92, None),
        (None, 1, 39, None,92 , None, 183, 830),
        (1, 49, 20, None, 180, 918, None, 9183),
    ]
    >>> group = [(0, 1), (1, 1), (2, 1), (3, 2), (4,2)]
    >>> data = pd.DataFrame(data)
    >>> group = pd.DataFrame(group, columns=['index', 'name'])
    >>> fillna_group_mean(data, group)

    """
    group_mean = group.groupby('name')['index'].apply(list).apply(lambda x: df.loc[x].mean()).reset_index()
    df_for_fillna = group.merge(group_mean, how='left').set_index('index').drop('name', axis=1)
    return df.fillna(df_for_fillna)
# coding=utf-8
import warnings
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def max_cutoff(arr, max_, alpha=0.75):
    # TODO finish max cutoff
    # ref: https://nlp.stanford.edu/pubs/glove.pdf

    if min(arr) < 0:
        warnings.warn('min of arr < 0')

    new_arr = np.array([1 if x >= max_ else pow(x * 1.0 / max_, alpha) for x in arr])
    return new_arr


def minmax():
    return MinMaxScaler()
# coding=utf-8

from ..utility import check_dtype
import numpy as np


def stat_dtype(arr, tsv=False):
    """
    Parameters
    ----------
    arr: array-like. data to be infered
    tsv: boolean, default False. return str: infer\tremarks

    Return
    ----------
    (infer, remarks) or infer\tremarks if tsv
    """

    remarks = []
    infer = []
    size = len(arr)

    all_digit = check_dtype.check_type(arr, (int, float, long))
    all_str = check_dtype.check_type(arr, str)
    all_digitable = check_dtype.all_digit_able(arr)
    all_intable = check_dtype.all_int_able(arr)

    if all_intable:
        # intable -> discrete
        infer.append('discrete_measurement')
        if all_digit:
            remarks.append('ints')
        elif all_str:
            remarks.append('ints as str')
        else:
            remarks.append('all int-able - mixed dtype')
        arr = [int(d) for d in arr]  # TODO: locale.atoi
        min_, max_ = min(arr), max(arr)
        if max_ <= size:
            # max<=size: possibly ordinal
            remarks.append('max<=size')
            infer.append('ordinal')
        nunique = len(np.unique(arr))
        if nunique <= 3:
            remarks.append('nunique={}'.format(nunique))
            if nunique==1:
                # nunique==1, constant
                infer.append('constant')
            else:
                # nunique <=3, it can be considered as nominal
                infer.append('nominal')
    elif all_digitable:
        # not intable but digitable -> digitable
        infer.append('continuous')
        if all_digit:
            remarks.append('digits(not all ints)')
        elif all_str:
            remarks.append('digits(not all ints) as str')
        else:
            remarks.append('all digit-able - mixed dtype')
    else:
        # not all digitable, containing at least one non-digit --> nominal
        infer.append('nominal')
        remarks.append('% digitable={:.2f}'.format(check_dtype.pcnt_digit_able(arr)))

    if tsv:
        return '{}\t{}'.format(', '.join(infer), ', '.join(remarks))
    return infer, remarks

# coding=utf-8
import locale
try:  # linux
    locale.setlocale(locale.LC_ALL, 'en_US.utf8')
except:  # windows
    locale.setlocale(locale.LC_ALL, 'english_USA')
import numpy as np


def check_type(a_list, types):
    """
    True if type of all elements isinstance of type or in types
    :param a_list: list or array
    :param types: type or types
    """
    return all(isinstance(x, types) for x in a_list)


def all_int(alist):
    return all(isinstance(x, (int, long)) for x in alist)


def all_digit(alist):
    return all(isinstance(x, (int, long, float, complex)) for x in alist)


def all_int_able(alist):
    """True if all elements can be cast as int by int(x) or locale.atoi(x)"""
    return all(int_able(x) for x in alist)


def all_digit_able(alist):
    """True if all elements can be cast as int by int(x) or locale.atoi(x)"""
    return all(digit_able(x) for x in alist)


def pcnt_digit_able(alist):
    """percentage of digitable elements"""
    return np.mean([digit_able(x) for x in alist])


def int_able(x):
    """True if x
        1. is float but float.is_integer()==True,
        2. can be cast as int by int(x) or locale.atoi(x)
    """
    if isinstance(x, float):
        return x.is_integer()
    try:
        int(x)
        return True
    except ValueError:
        try:
            locale.atoi(x)
            return True
        except ValueError:
            return False


def digit_able(x):
    """True if x can be cast into float by float(x) or locale.atof(x)"""
    try:
        float(x)
        return True
    except ValueError:
        try:
            locale.atof(x)
            return True
        except ValueError:
            return False

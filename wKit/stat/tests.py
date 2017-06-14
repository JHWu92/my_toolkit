# coding=utf-8
from scipy import stats


def significant_level(p):
    if p <= 0.001:
        return '***'
    if p <= 0.01:
        return '**'
    if p <= 0.05:
        return '*'
    return ''


def krutest(list_obs):
    try:
        stat, p = stats.kruskal(*list_obs)
    except ValueError:
        stat, p = None, None
    return stat, p


def f_oneway(list_obs):
    stat, p = stats.f_oneway(*list_obs)
    return stat, p


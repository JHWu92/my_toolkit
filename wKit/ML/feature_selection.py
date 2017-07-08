# coding=utf-8
import numpy as np
from dprep import discretize_features
from skfeature.function.information_theoretical_based import MRMR
from sklearn.feature_selection import RFECV, VarianceThreshold
from sklearn.linear_model import LinearRegression, RandomizedLasso, RandomizedLogisticRegression
from sklearn.svm import LinearSVC
import pandas as pd
import datetime
def help():
    import types
    for name, obj in globals().items():
        if name == 'help':
            continue
        if isinstance(obj, types.FunctionType):
            print('Function:', name)
            print(obj.__doc__)


def fselect(x, y, name, **kwargs):
    """
    API for feature selection. Choose fucntion by name
    :param: name choices:
        classification: stability_logistic, rfecv_linsvc, mrmr
        regression: stability_lasso, rfecv_linreg
        unsupervised: has_value_thres, var_thres
    :param kwargs:
        random_state: for train_test_split, default 0
        cv: k-fold cv, for rfecv_* feature selection
        n_jobs: num of threads for rfecv_* feature selection
        param: param for some feature selection model. rfecv_*, stability_*
        thres: for thres based selection: has_value_thres, var_thres
        verbose: True to print(timestamp, default False
    :param kwargs

    Return array of boolean, True means important.
    """
    if isinstance(x, pd.DataFrame):
        x = x.values
    if isinstance(y, pd.Series):
        y = y.values
    if kwargs.get('verbose', False):
        print(datetime.datetime.now(), 'feature selection with', name)
    return globals()[name](x=x, y=y, **kwargs)


def has_value_thres(x, **kwargs):
    """
    unsupervised feature selection. Keep features whose percentage of non-NA is larger than thres.
    I.e. thres = 0 means keeping features of which at least one sample has value

    Parameters
    ----------
    x: np 2d array
    thres: has value threshold. Default 0.1.

    Return
    ----------
    array of boolean, True means important.
    """
    thres = kwargs.get('thres', 0.1)
    has_value_percentage = (~np.isnan(x)).mean(axis=0)
    return has_value_percentage > thres


def var_thres(x, **kwargs):
    """
    unsupervised feature selection. Keep features with variance larger than thres.
    Parameters
    ----------
    x: np 2d array
    thres: variance threshold. Default 0.0, i.e. remove features with the same value for all samples.

    Return
    ----------
    array of boolean, True means important.
    """

    thres = kwargs.get('thres', 0.0)
    selector = VarianceThreshold(threshold=thres)
    selector.fit(x)
    return selector.get_support()


def stability_lasso(x, y, **kwargs):
    """
    Stability selection for regression problem with randomized lasso.
    :param: x: np 2d array
    :param: param for RandomizedLasso. If provided, RandomizedLasso.set_param(param)
    :return: array of boolean, True means important.
    """
    rl = RandomizedLasso()
    if 'param' in kwargs:
        rl.set_params(**kwargs['param'])
    rl.fit(x, y)
    return rl.get_support()


def stability_logistic(x, y, **kwargs):
    """
    Stability selection for binary or multi classification problem with randomized logistic regression.
    :param: x: np 2d array
    :param: param for RandomizedLogisticRegression. If provided, RandomizedLogisticRegression.set_param(param)
    :return: array of boolean, True means important.
    """
    rlr = RandomizedLogisticRegression(n_jobs=kwargs.get('n_jobs', 4))
    if 'param' in kwargs:
        rlr.set_params(**kwargs['param'])
    rlr.fit(x, y)
    return rlr.get_support()


def rfecv_linreg(x, y, **kwargs):
    """
    RFECV for regression problem with linear regression.
    Optimum number of features is decided through cv.
    Parameters
    ----------
    x: np 2d array
    cv: k-fold cross validation, default 5
    n_jobs: parallel cpu threads, default 4

    Return
    ----------
    array of boolean, True means important.
    """
    cv = kwargs.get('cv', 5)
    n_jobs = kwargs.get('n_jobs', 4)
    lr = LinearRegression()
    rfe = RFECV(lr, step=1, cv=cv, n_jobs=n_jobs)
    rfe.fit(x, y)
    return rfe.support_


def rfecv_linsvc(x, y, **kwargs):
    """
    RFECV for binary or multi classification problem with linear SVC.
    Optimum number of features is decided through cv.
    Parameters
    ----------
    x: np 2d array
    cv: k-fold cross validation, default 5
    n_jobs: parallel cpu threads, default 4
    param: for linear svc. If provided, linear SVC set param(param)

    Return
    ----------
    array of boolean, True means important.
    """
    cv = kwargs.get('cv', 5)
    n_jobs = kwargs.get('n_jobs', 4)
    lscv = LinearSVC()
    if 'param' in kwargs:
        lscv.set_params(**kwargs['param'])
    rfe = RFECV(lscv, step=1, cv=cv, n_jobs=n_jobs)
    rfe.fit(x, y)
    return rfe.support_


def mrmr(x, y, **kwargs):
    """
    mRMR for classification. Features are processed by dpred.discretize_features.
    :param: x: np 2d array
    :return: array of boolean, True means important
    """
    n_features = x.shape[1]
    discrete_x = discretize_features(x)
    rank = MRMR.mrmr(discrete_x, y)
    support = [True if i in rank else False for i in range(n_features)]
    return np.array(support)

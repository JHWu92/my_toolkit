# coding=utf-8
from dprep import discretize_features
from skfeature.function.information_theoretical_based import MRMR
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LinearRegression, RandomizedLasso, RandomizedLogisticRegression
from sklearn.svm import LinearSVC


def help():
    import types
    for name, obj in globals().items():
        if name == 'help':
            continue
        if isinstance(obj, types.FunctionType):
            print 'Function:', name
            print obj.__doc__


def main(x, y, name, **kwargs):
    """
    API for feature selection. Choose fucntion by name
    Return array of boolean, True means important.
    """
    return globals()[name](x, y, **kwargs)


def stability_lasso(x, y, **kwargs):
    """
    Stability selection for regression problem with randomized lasso.
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


def mrmr(x, y):
    """
    mRMR for classification. Features are processed by dpred.discretize_features.
    :return: array of boolean, True means important
    """
    import numpy as np
    n_features = x.shape[1]
    discrete_x = discretize_features(x)
    rank = MRMR.mrmr(discrete_x, y)
    support = [True if i in rank else False for i in range(n_features)]
    return np.array(support)

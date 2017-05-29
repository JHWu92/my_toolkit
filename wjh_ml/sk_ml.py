# coding=utf-8
from datetime import datetime as dtm
import os

from sklearn import linear_model, svm, tree, ensemble, neural_network, naive_bayes
from sklearn.metrics import mean_squared_error, f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def sk_models(reg=True, cls=True, stoplist=('SVM', 'SVR', 'GDBreg', 'GDBcls')):
    """
    return sk models with names by regression and/or classification.
    default stoplist is ('SVM', 'SVR', 'GDBreg', 'GDBcls') because they are too slow
    """
    reg_models = {
        'ols'      : linear_model.LinearRegression(),
        'ridge'    : linear_model.Ridge(),
        'lasso'    : linear_model.Lasso(),
        'DTreg'    : tree.DecisionTreeRegressor(),
        'RFreg'    : ensemble.RandomForestRegressor(),
        'ADAreg'   : ensemble.AdaBoostRegressor(),
        'BAGreg'   : ensemble.BaggingRegressor(),
        'GDBreg'   : ensemble.GradientBoostingRegressor(),
        'SVR'      : svm.SVR(),
        'linearSVR': svm.LinearSVR(),
        'MLPreg'   : neural_network.MLPRegressor(),
    }

    cls_models = {
        'logistics': linear_model.LogisticRegression(),
        'DTcls'    : tree.DecisionTreeClassifier(),
        'RFcls'    : ensemble.RandomForestClassifier(),
        'ADAcls'   : ensemble.AdaBoostClassifier(),
        'BAGcls'   : ensemble.BaggingClassifier(),
        'GDBcls'   : ensemble.GradientBoostingClassifier(),
        'SVM'      : svm.SVC(),
        'linearSVM': svm.LinearSVC(),
        'MLPcls'   : neural_network.MLPClassifier(),
        'GNBcls'   : naive_bayes.GaussianNB(),
    }

    models = {}
    if reg:
        for name in stoplist: reg_models.pop(name, None)
        models['reg'] = reg_models
    if cls:
        for name in stoplist: cls_models.pop(name, None)
        models['cls'] = cls_models
    return models


# ################################################
# helper
# ################################################

def bounded_round(arr, mini, maxi):
    arr_round = arr.round()
    arr_round[arr_round < mini] = mini
    arr_round[arr_round > maxi] = maxi
    return arr_round


# ################################################
# pre-processing
# ################################################

def fillna(df, how='mean'):
    """df is the dataset
    """
    if how == 'mean':
        return df.fillna(df.mean())
    return df.fillna(how)


def scaler_by_name(name):
    """return sklearn scaler by name (MinMaxScaler,)"""
    norm_choices = {'MinMaxScaler': MinMaxScaler()}
    return norm_choices[name]


# ################################################
# Grid search cross validation
# ################################################

def grid_cv_default_params():
    # GDBreg's parameters are deliberately cut down.
    params_gdb = {'n_estimators': [10, 50, 100], 'max_features': [0.1, 0.5, 1.], 'learning_rate': np.logspace(-4, 1, 3),
                  'max_depth'   : [3, 10, 50]},
    params_rf = {'n_estimators': [10, 30, 50, 100, 256, 500], 'max_features': [0.1, 0.3, 0.5, 1.]}
    params_ada = {'n_estimators': [10, 30, 50, 100, 256, 500], 'learning_rate': np.logspace(-4, 1, 5)}
    params_bag = {'n_estimators': [10, 30, 50, 100, 256, 500], 'max_features': [0.4, 0.7, 1.0]}

    # SVM/SVR is way too slow
    c_s = np.logspace(-4, 2, 3)
    gamma_s = [1e-5, 1e-3, 1e-1]

    params_svm = [
        {'kernel': ['rbf'], 'C': c_s, 'gamma': gamma_s},
        {'kernel': ['sigmoid'], 'C': c_s, 'gamma': gamma_s},
        {'kernel': ['poly'], 'C': c_s, 'gamma': gamma_s, 'degree': [3]},
    ]

    params_svr = [
        {'kernel': ['rbf'], 'C': c_s, 'gamma': gamma_s},
        {'kernel': ['sigmoid'], 'C': c_s, 'gamma': gamma_s},
        {'kernel': ['poly'], 'C': c_s, 'gamma': gamma_s, 'degree': [3]},
    ]

    params_reg = {
        # regression
        'ols'      : {},
        'ridge'    : {'alpha': np.logspace(0, 2, 10)},
        'lasso'    : {'alpha': np.logspace(0, 2, 10)},
        'DTreg'    : {'max_depth': [3, 5, 10, 30, 50], 'max_features': [0.1, 0.3, 0.5, 1.]},
        'RFreg'    : params_rf,
        'ADAreg'   : params_ada,
        'BAGreg'   : params_bag,
        'GDBreg'   : params_gdb,
        'SVR'      : params_svr,
        'linearSVR': {'C': c_s, 'loss': ['epsilon_insensitive', 'squared_epsilon_insensitive'], 'epsilon': [0, 0.1, 1]},
        'MLPreg'   : {'hidden_layer_sizes': [(5, 2), (20, 5), (100, 20), (100, 20, 5)],
                      'learning_rate'     : ['constant', 'adaptive'], 'max_iter': [10000]},
    }

    params_cls = {
        'logistics': {'C': np.logspace(-4, 2, 4), 'penalty': ['l1', 'l2']},
        'DTcls'    : {'max_depth': [3, 5, 10, 30, 50], 'max_features': [0.1, 0.3, 0.5, 1.],
                      'criterion': ['gini', 'entropy']},
        'RFcls'    : params_rf,
        'ADAcls'   : params_ada,
        'BAGcls'   : params_bag,
        'GDBcls'   : params_gdb,
        'SVM'      : params_svm,
        'linearSVM': {'C': c_s, 'loss': ['hinge', 'squared_hinge']},
        'MLPcls'   : {'hidden_layer_sizes': [(5, 2), (20, 5), (100, 20), (100, 20, 5)],
                      'learning_rate'     : ['constant', 'adaptive'], 'max_iter': [10000]},
        'GNBcls'   : {},
    }

    return {'cls': params_cls, 'reg': params_reg}


def grid_cv_a_model(x, y, model, param, kind, name, path='', n_jobs=4, cv=5, verbose=False, redo=False, save_res=True,
                    fit_when_load=True):
    """
    cv a model, return cv result of the best model
    if model result exists, load the existing best parameters setting, but without grid_cv_time
    when loading model, run empty model.fit(x,y) if fit_when_load=True
    """
    scoring = 'neg_mean_squared_error' if kind == 'reg' else 'f1_weighted'
    path_model_res = os.path.join(path, 'cv_%d_model_%s.csv' % (cv, name))

    if os.path.exists(path_model_res) and not redo:
        print 'loading existing model', kind, name
        model_res = pd.read_csv(path_model_res, index_col=0)
        best = model_res.iloc[0].to_dict()
        param = eval(best['params'])
        model.set_params(**param)
        if fit_when_load:
            if verbose:
                print 'fitting model', kind, name
            model.fit(x, y)
        result = {'grid_cv_time' : None, 'score': scoring, 'model_name': name, 'kind': kind,
                  'mean_test'    : best['mean_test_score'], 'mean_train': best['mean_train_score'],
                  'mean_fit_time': best['mean_fit_time'], 'best_params': param, 'best_model': model,
                  }
        print 'loaded existing result for model:', name
        return result

    sub_start = dtm.now()
    print sub_start, 'CVing: kind = {}, model = {}'.format(kind, name)
    clf = GridSearchCV(model, param, n_jobs=n_jobs, cv=cv, scoring=scoring)
    clf.fit(x, y)
    sub_end = dtm.now()

    df = pd.DataFrame(clf.cv_results_).sort_values(by='mean_test_score', ascending=False)
    if save_res:
        df.to_csv(path_model_res)

    test_score, train_score, fit_time = df[['mean_test_score', 'mean_train_score', 'mean_fit_time']].values[0]

    if verbose:
        print 'score: %s, best test = %.3f, train = %.3f, mean_fit_time = %f' % (
            scoring, test_score, train_score, fit_time)
        print 'best params', clf.best_params_
        print sub_end, sub_end - sub_start
        print

    result = {
        'grid_cv_time' : sub_end - sub_start,
        'score'        : scoring,
        'model_name'   : name,
        'kind'         : kind,
        'mean_test'    : test_score,
        'mean_train'   : train_score,
        'mean_fit_time': fit_time,
        'best_params'  : clf.best_params_,
        'best_model'   : clf.best_estimator_,
    }
    return result


def grid_cv_models(x, y, models, params, path='', n_jobs=4, cv=5, save_res=True, redo=False, verbose=False,
                   fit_when_load=True):
    """
    regression model is evaluated by neg_mean_squared_error
    classification model is evaluated by f1_weighted
    iterate over models' keys, get tuning parameters based on key, if no matched paramters, that model will be skipped
    if not redo and the result exists, optimum parameters will be loaded using model.set_params(**loaded)
    when loading model, run empty model.fit(x,y) if fit_when_load=True
    :return:
        index: (kind, name);
        each line is the best parameters for that model;
        type of column "best_model" is sklearn models.
    """
    path_cv_best = os.path.join(path, 'cv_%d_best_models.csv' % cv)

    # if cv best result exists and not redeo, load existing parameters
    if os.path.exists(path_cv_best) and not redo:
        loaded_df_cv_res = pd.read_csv(path_cv_best, index_col=[0, 1])
        best_models = []
        for (kind, name), row in loaded_df_cv_res.iterrows():
            param = eval(row.best_params)
            model = models[kind][name]
            model.set_params(**param)
            if fit_when_load:
                if verbose: print 'fitting model', kind, name
                model.fit(x, y)
            best_models.append(model)

        loaded_df_cv_res.best_model = best_models
        print 'loaded existing cv-ed best parameters'
        return loaded_df_cv_res

    # redo or result not exists
    cv_results = []
    start = dtm.now()
    for kind in ['reg', 'cls']:
        if kind not in models:
            continue
        for name, model in models[kind].items():
            if name not in params[kind]:
                print 'model', name, 'doesnt have tuning params, use {} instead'
                param = {}
            else:
                param = params[kind][name]

            result = grid_cv_a_model(x, y, model, param, kind, name,
                                     path=path, n_jobs=n_jobs, cv=cv, verbose=verbose, redo=redo, save_res=save_res,
                                     fit_when_load=fit_when_load)
            cv_results.append(result)

    end = dtm.now()
    print 'finished CV', end, end - start

    df_cv = pd.DataFrame(cv_results).set_index(['kind', 'model_name'])
    if save_res:
        df_cv.to_csv(path_cv_best)

    return df_cv


# ################################################
# Evaluation
# Evaluators for different prediction tasks
# ################################################

def evaluate_grid_cv(df_cv, train_x, train_y, test_x, test_y, evaluator, path='', cv=5, save_res=True):
    """
    This function is major for evaluate result from grid_cv_models. But it can be also used to evaluate df containing
    different models. In this case, set save_res=False.

    Parameters
        df_cv: results from :func:: grid_cv_models(). index=(kind, name), 'best_model' is in columns.
        evaluator: such as :func:: evaluator_scalable_cls()
        save_res: True->save to path/cv_%d_best_models_evaluation.csv % cv
    Return
        evaluation result as pd.DF, columns are defined by evaluator
    """
    print 'evaluating grid cv'
    results = {}
    for (kind, name), model in df_cv.best_model.iteritems():
        results[kind, name] = evaluator(model, train_x, train_y, test_x, test_y)

    df_results = pd.DataFrame(results).T
    if 'test_f1' in df_results.columns:
        df_results.sort_values(by='test_f1', ascending=False, inplace=True)

    if save_res:
        df_results.to_csv(os.path.join(path, 'cv_%d_best_models_evaluation.csv' % cv))

    return df_results


def evaluator_scalable_cls(model, train_x, train_y, test_x, test_y):
    """
    Evaluator for scalable classification. E.g. Ys are 1, 2, 3, 4
    Both regression and classification will be used.
    prediction by regression will be round up (bounded by max and min of Ys) as a class label
    :return: metrics: mse, accuracy and weighted f1, for both train and test
    """
    min_y, max_y = train_y.min(), train_y.max()

    model.fit(train_x, train_y)

    train_pred = model.predict(train_x)
    train_pred_round = bounded_round(train_pred, min_y, max_y)

    mse_train = mean_squared_error(train_y, train_pred)
    acc_train = accuracy_score(train_y, train_pred_round)
    f1_train = f1_score(train_y, train_pred_round, average='weighted')

    test_pred = model.predict(test_x)
    test_pred_round = bounded_round(test_pred, min_y, max_y)

    mse_test = mean_squared_error(test_y, test_pred)
    acc_test = accuracy_score(test_y, test_pred_round)
    f1_test = f1_score(test_y, test_pred_round, average='weighted')

    result = {
        'train_f1' : f1_train,
        'train_acc': acc_train,
        'train_mse': mse_train,
        'test_f1'  : f1_test,
        'test_acc' : acc_test,
        'test_mse' : mse_test,
    }
    return result


def confusion_matrix_as_df(fitted_model, x, y, labels=None):
    """
    build a confusion matrix between y and fitted_model.predict(x)

    parameters:
        fitted_model: model from sklearn. It should already perform fit(train_x, train_y)
        x,y: features and true labels
        labels: index and column names of the return pd.df. If None, pd.unique(y) will be used

    return：
        confusion matrix in the form of pd.Dataframe
    """

    pred_y = fitted_model.predict(x)
    if labels is None:
        labels = pd.unique(y)
    cfsn = confusion_matrix(y, pred_y, labels=labels)
    return pd.DataFrame(cfsn, columns=labels, index=labels)


# ################################################
# Analysis of predictor
# Visualization cross validation result
# ################################################

def vis_evaluation(path, cv):
    df_eval = pd.read_csv(os.path.join(path, 'cv_%d_best_models_evaluation.csv' % cv))
    return df_eval.plot()


def vis_grid_cv_one_model(fn):
    df = pd.read_csv(fn, index_col=0)
    return df[['mean_test_score', 'mean_train_score']].boxplot()


def show_important_features(tree_model, name="", top=None, labels=None, show_plt=True, set_std=True):
    """
    Format tree_models feature importance as pd.df, along with a bar plot with error bar.

    TODO:
        the shape of estimators of gradient boosting is [n_estimator, n_classes(binary=1)].
        not sure the difference per row. Right now just flatten all estimators to calculate std.

    :param tree_model: tree like model, mostly models in sklearn.ensemble.
    :param name: name to be shown in plot title
    :param top: show top most important features. Default None, showing all features
    :param labels: labels for features. Default None, range(len(tree_model.feature_importances_))
    :param show_plt: Default True, visualize importance and std as error bar with maplotlib.pyplot.
    :return: pd.df, columns = importance, label, std
    """
    importances = tree_model.feature_importances_
    feature_size = len(importances)
    if hasattr(tree_model, 'estimators_') and set_std:
        # TODO: is it reasonable to flatten gradient boosting's estimators?
        estimators = tree_model.estimators_
        if hasattr(estimators, 'flatten'):
            estimators = estimators.flatten()
        std = np.std([tree.feature_importances_ for tree in estimators], axis=0)
    else:
        std = np.zeros(feature_size)
    if labels is None:
        labels = [i for i in range(len(importances))]
    top = min(feature_size, top) if top is not None else feature_size

    imp = pd.DataFrame(zip(importances, labels, std), columns=['importance', 'label', 'std'])
    imp.sort_values('importance', ascending=False, inplace=True)
    imp = imp[:top]

    if show_plt:
        title = "%s Feature importances - top %d / %d %s" % (
            name, top, len(importances), 'no errbar' if not set_std else 'with errbar')
        imp_plt = imp.set_index('label')
        imp_plt.importance.plot(kind='barh', xerr=imp_plt['std'], title=title, figsize=(10, 7))

    return imp

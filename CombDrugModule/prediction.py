from sklearn import model_selection, metrics, ensemble, feature_selection
import scipy as sp
import catboost as cb
import lightgbm as lgb
import xgboost as xgb
import pandas as pd

from scipy.stats.stats import pearsonr
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer


###############
# Regressors #
##############
REGRESSORS = {
    'CatBoost':{
        'name':'Gradient Boosting on Decision Trees',
        'class':cb.CatBoostRegressor,
        'default_params':{
            "loss_function":'RMSE',
            "eval_metric":'RMSE',
            'iterations':1000,
            'depth':6,
            "bootstrap_type":"MVS",
            "subsample": 0.66,
            "colsample_bylevel":1.0,
            "task_type":"CPU",
            "random_state":120,
            "early_stopping_rounds":10,
            "thread_count":4,
            "verbose":False
        },
        'param_for_grids':{
            'iterations':[500, 1000, 2000],
            #'iterations':[2000, 5000],
            'depth':[4, 6, 8, 10], 
            #'depth':[8, 10, 14], 
            "subsample": [0.5, 0.66, 1.0],
            "colsample_bylevel":[0.3, 0.5, 1.0]
        },
        'param_for_random':{
            'iterations': sp.stats.randint(10, 1001),
            'depth': sp.stats.randint(3, 16),
            'learning_rate': sp.stats.uniform(0.01, 0.99),
            'random_strength': sp.stats.uniform(1e-9, 21),
            'bagging_temperature': sp.stats.uniform(0.05, 0.94),
            'border_count': sp.stats.randint(1, 256),
            'l2_leaf_reg': sp.stats.randint(2, 15)
        }
    },
    'HistGradientBoostingRegressor':{
        'name':'Histogram-based Gradient Boosting Regression Tree',
        'class':ensemble.HistGradientBoostingRegressor,
        'default_params':{
            "random_state":120,
            "early_stopping":True,
            "verbose":False
        },
        'param_for_bayes':{
            'learning_rate': Real(0.01, 1.0, 'log-uniform'),
            'max_iter':Integer(10, 1000),
            'max_bins':Integer(2, 255),
            'max_depth':Integer(3, 15),
            'l2_regularization':Real(1e-7, 10, 'log-uniform')
        },
        'param_for_random':{
            'learning_rate': sp.stats.uniform(0.01, 0.99),
            'max_iter': sp.stats.randint(10, 1001),
            'max_bins': sp.stats.randint(2, 256),
            'max_depth': sp.stats.randint(3, 16),
            'l2_regularization':sp.stats.uniform(1e-7, 9.99)
        }
    },
    'LightGBM':{
        'name':'Light Gradient Boosting Machine',
        'class':lgb.LGBMRegressor,
        'default_params':{
            "boosting_type":'gbdt',
            "objective":'regression',
            "metric":'rmse',
            "random_state":120,
            "n_jobs":1
        },
        'param_for_bayes':{
            'n_estimators': Integer(10, 1000),
            'num_leaves': Integer(10, 100),
            'max_depth':Integer(3, 20),
            'learning_rate': Real(0.01, 0.99, 'log-uniform'),
            'colsample_bytree': Real(0.05, 1.0),
        },
        'param_for_random':{
            'n_estimators': sp.stats.randint(10, 1001),
            'num_leaves': sp.stats.randint(10, 101),
            'max_depth': sp.stats.randint(3, 21),
            'learning_rate': sp.stats.uniform(0.01, 0.99),
            'colsample_bytree': sp.stats.uniform(0.05, 0.95),
        }
    },
    'XGBoost':{
        'name': 'eXtreme Gradient Boosting',
        'class':xgb.XGBRegressor,
        'default_params':{
            "objective":"reg:squarederror",
            "eval_metric":"rmse",
            "n_estimators":100,
            "max_depth":6, 
            "subsample": 1.0,
            "colsample_bylevel":1.0,
            "n_jobs":4,
            #"early_stopping_rounds":10,
            "random_state":120,
            #"verbose":False
        },
        'param_for_grids':{
            'n_estimators':[100, 500, 1000, 2000],
            'max_depth':[4, 6, 10], 
            "subsample": [0.5, 0.66, 1.0],
            "colsample_bylevel":[0.3, 0.5, 1.0]
        },
        'param_for_random':{
            'n_estimators': sp.stats.randint(10, 1001),
            'max_depth': sp.stats.randint(3, 21),
            'learning_rate': sp.stats.uniform(0.01, 0.99),
            'colsample_bytree': sp.stats.uniform(0.05, 0.95),
        }
    },
    'RandomForest':{
        'name': 'Random Forest',
        'class': ensemble.RandomForestRegressor,
        'default_params':{
            "bootstrap":True,
            "warm_start":True, 
            "random_state":120,
            "max_depth":None,
            "max_features":"auto",
            "n_jobs":4,
            "verbose":0
        },
        'param_for_grids':{
            'n_estimators': [100, 1000, 2000, 3000],
            'max_features':  [0.3, 0.5, 1.0],
        },
        'param_for_random':{
            'n_estimators': sp.stats.randint(10, 501),
            'max_depth': sp.stats.randint(3, 16),
            #'max_features':  sp.stats.uniform(0.2, 0.97),
            'max_features':  sp.stats.uniform(0.1, 0.49),
        }
    }
}

def getRegressor(regressor_name, best_params=None, regressor_params=None):
    
    params = REGRESSORS[regressor_name]['default_params']
    if best_params:
        params.update(REGRESSORS[regressor_name]['best_params'])
    
    if regressor_params:
        params.update(regressor_params)
    
    return REGRESSORS[regressor_name]['class'](**params)

#########
# Folds #
#########
FOLDS = {
    'KFold': {
        'name':'k-fold',
        'class':model_selection.KFold,
        'params': {
            'n_splits': 5,
            'shuffle': True,
            'random_state': 120,
        }
    },
    'StratifiedKFold': {
        'name': 'Stratified K-Folds',
        'class': model_selection.StratifiedKFold,
        'params': {
            'n_splits': 5,
            'shuffle': True,
            'random_state': 120,
        }
    },
    'GroupKFold': {
        'name':'GroupKFold',
        'class':model_selection.GroupKFold,
        'params': {
            'n_splits': 5,
        }
    }
}


def getFolds(folds_name, X, y, folds_params=None, group=None):

    #Get default params
    params = FOLDS[folds_name]['params']

    # Update params if given
    if folds_params:
        params.update(folds_params)

    # Set default params depending on y and labels
    if folds_name == 'KFold':
        return FOLDS['KFold']['class'](**params).split(X)
    elif folds_name == 'StratifiedKFold':
        return FOLDS['StratifiedKFold']['class'](**params).split(X, y)
    elif folds_name == 'GroupKFold':
        return FOLDS['GroupKFold']['class'](**params).split(X, y, group)
    
def scoring_pearson_r(y_true, y_pred):
    return pearsonr(y_true, y_pred)[0]

def getScoreFunc(score_params):

    if score_params == "AUC":
        return metrics.make_scorer(metrics.roc_auc_score, needs_proba=True)
    elif score_params == "ACC":
        return metrics.make_scorer(metrics.accuracy_score)
    elif score_params == "MCC":
        return metrics.make_scorer(metrics.matthews_corrcoef)
    elif score_params == "RMSE":
        return metrics.make_scorer(metrics.mean_squared_error, greater_is_better=False, squared=False)
    elif score_params == "MSE":
        return metrics.make_scorer(metrics.mean_squared_error, greater_is_better=False, squared=True)
    elif score_params == "r2":
        return metrics.make_scorer(metrics.r2_score, greater_is_better=True)
    elif score_params == "pearsonr":
        return metrics.make_scorer(scoring_pearson_r, greater_is_better=True)
    
def getXy(data, feature_prefix='feature_', y_col_name='Loewe'):
    """Get features matrix X and label y from pandas"""
    X = data.filter(regex="^{}".format(feature_prefix), axis=1).values
    y = data[y_col_name].values
    return X, y

def selKFeatures(data, score_name, sel_func = "MI", keep_threshold = 100):
    # Get raw columns
    raw_cols = data.columns.drop("score_{}".format(score_name))
    # Get X, y
    X, y = getXy(data, feature_prefix='feature_', 
                 y_col_name="score_{}".format(score_name))
    # Feature selection methods
    if sel_func == "MI":
        # Estimate mutual information for a continuous target variable.
        skb = feature_selection.SelectKBest(score_func=feature_selection.mutual_info_regression,
                                            k=keep_threshold)
    elif sel_func == "Fvalue":
        # Univariate linear regression tests returning F-statistic and p-values.
        skb = feature_selection.SelectKBest(score_func=feature_selection.f_regression,
                                            k=keep_threshold)
    
    sel_skb = skb.fit(X, y)
    remain_cols = raw_cols[sel_skb.get_support()].append(pd.Index(["score_{}".format(score_name)]))
    
    data_sel = data[remain_cols]
    
    return data_sel

def selKFeatures_old(data, score_name, r_threshold = 0.1, keep_threshold = 100):
    # Get raw columns
    raw_cols = data.columns.drop("score_{}".format(score_name))
    # Get X, y
    X, y = getXy(data, feature_prefix='feature_', 
                 y_col_name="score_{}".format(score_name))
    
    # Compute Pearson’s r for each features and the target.
    r_reg_res = feature_selection.r_regression(X, y)
    remain_cols = raw_cols[abs(r_reg_res) >= r_threshold].append(pd.Index(["score_{}".format(score_name)]))
    
    data_rm_r = data[remain_cols]
    cols_rm_r = data_rm_r.columns.drop("score_{}".format(score_name))

    del data
    
    if len(cols_rm_r) < keep_threshold:
        return data_rm_r
    else:
        # Get X, y
        X, y = getXy(data_rm_r, feature_prefix='feature_',
                     y_col_name="score_{}".format(score_name))
        # Estimate mutual information for a continuous target variable.
        skb = feature_selection.SelectKBest(score_func=feature_selection.mutual_info_regression,
                                            k=keep_threshold)
        sel_skb = skb.fit(X, y)
        remain_cols = cols_rm_r[sel_skb.get_support()].append(pd.Index(["score_{}".format(score_name)]))
        
        data_sel = data_rm_r[remain_cols]
        
        del data_rm_r
        
        return data_sel
    
def bayesParameterSearch(data, regressor_name, n_iter, n_jobs, random_num, 
                         feature_prefix='feature_', y_col_name='score_Loewe',
                         folds_name = "KFold", score_params = "RMSE",
                         param_for_search=None, folds_params=None, 
                         group_name=None):
    
    # Get X, y
    X, y = getXy(data, feature_prefix=feature_prefix, y_col_name=y_col_name)
    
    # Get regressor
    estimator = getRegressor(regressor_name)
    
    # Get folds
    folds = getFolds(folds_name, X, y)
    
    # Use scorer
    scorer_func = getScoreFunc(score_params)
    
    # Get parameters distribution for estimator
    if param_for_search is None:
        param_distributions = REGRESSORS[regressor_name]['param_for_bayes']
    else:
        param_distributions = param_for_search
        
    # Setting up BayesSearchCV
    opt = BayesSearchCV(estimator,
                        param_distributions,
                        scoring=scorer_func,
                        cv=folds,
                        n_iter=n_iter,
                        n_jobs=n_jobs,
                        return_train_score=False,
                        refit=False,
                        optimizer_kwargs={'base_estimator': 'GP'},
                        random_state=random_num, verbose=1)
    
    opt.fit(X, y)
    
    return opt

def gridParameterSearch(data, regressor_name, n_jobs,
                        feature_prefix='feature_', y_col_name='score_Loewe',
                        folds_name = "KFold", score_params = "RMSE",
                        param_for_search=None, folds_params=None, 
                        group_name=None):
    # Get X, y
    X, y = getXy(data, feature_prefix=feature_prefix, y_col_name=y_col_name)
    
    # Get regressor
    estimator = getRegressor(regressor_name)
    
    # Get folds
    folds = getFolds(folds_name, X, y, folds_params)
    
    # Use scorer
    scorer_func = getScoreFunc(score_params)
    
    # Get parameters distribution for estimator
    if param_for_search is None:
        param_distributions = REGRESSORS[regressor_name]['param_for_grids']
    else:
        param_distributions = param_for_search
    
    # Random search
    search = model_selection.GridSearchCV(estimator,
                                          param_grid = param_distributions,
                                          scoring=scorer_func,
                                          n_jobs=n_jobs,
                                          refit=False,
                                          cv=folds,
                                          return_train_score=False,
                                          verbose=1)

    search.fit(X, y)
    
    return search

def randomParameterSearch(data, regressor_name, n_iter, n_jobs, random_num,
                          feature_prefix='feature_', y_col_name='score_Loewe',
                          folds_name = "KFold", score_params = "RMSE",
                          param_for_search=None, folds_params=None, 
                          group_name=None):
    # Get X, y
    X, y = getXy(data, feature_prefix=feature_prefix, y_col_name=y_col_name)
    
    # Get regressor
    estimator = getRegressor(regressor_name)
    
    # Get folds
    folds = getFolds(folds_name, X, y, folds_params)
    
    # Use scorer
    scorer_func = getScoreFunc(score_params)
    
    # Get parameters distribution for estimator
    if param_for_search is None:
        param_distributions = REGRESSORS[regressor_name]['param_for_random']
    else:
        param_distributions = param_for_search
    
    # Random search
    search = model_selection.RandomizedSearchCV(estimator,
                                                param_distributions,
                                                scoring=scorer_func,
                                                cv=folds,
                                                n_iter=n_iter,
                                                n_jobs=n_jobs,
                                                return_train_score=False,
                                                random_state=random_num,
                                                verbose=1)

    search.fit(X, y)
    
    return search

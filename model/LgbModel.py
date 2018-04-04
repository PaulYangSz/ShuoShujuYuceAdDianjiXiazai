"""
Use LightGBoost to predict
"""
import platform
from functools import reduce

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import gc
import lightgbm as lgb
import time
import os
from pprint import pprint
import hyperopt
from hyperopt import hp, fmin, tpe, Trials
from hyperopt import space_eval
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedKFold, train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score
import logging
import logging.config

from model.DataReader import DataReader, timer


if platform.system() == 'Windows':
    N_CORE = 1
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    # 有中文出现的情况，需要u'内容'


def start_logging():
    pd.set_option('display.max_rows', 100)
    pd.set_option('display.max_columns', 100)
    pd.set_option('display.width', 10000)
    # 加载前面的标准配置
    from logging_config import ConfigLogginfDict
    logging.config.dictConfig(ConfigLogginfDict(__file__).LOGGING)
    # 获取loggers其中的一个日志管理器
    logger = logging.getLogger("default")
    logger.info('\n\n#################\n~~~~~~Start~~~~~~\n#################')
    print(type(logger))
    return logger


if 'Logger' not in dir():
    Logger = start_logging()


class LgbHpParamSelect():
    classifier_score = "roc_auc"
    rand_state = 20180401
    space_dict = dict()

    def __init__(self, space_name: str):
        if space_name == "default":
            self.name = space_name
            self.space_dict = {
                'num_leaves': 12,  # 2 + hp.randint('num_leaves', 20),
                'max_depth': -1,  # hp.choice('max_depth', [-1])
                'learning_rate': 0.02,  # hp.quniform('learning_rate', 0.001, 0.04, 0.001),
                'n_estimators': 60,  # (1 + hp.randint('n_estimators', 9)) * 10,
                'min_split_gain': 0.0,  # hp.quniform('min_split_gain', 0.0, 0.5, 0.01),
                'min_child_weight': 0.01,  # hp.quniform('min_child_weight', 0.001, 0.1, 0.001),
                'min_child_samples': 1,  # 1 + hp.randint('min_child_samples', 10),
                'subsample': 0.9,  # hp.quniform('subsample', 0.5, 1.0, 0.01),
                'subsample_freq': 1,  # 1 + hp.randint('subsample_freq', 50),  # frequency for bagging
                'colsample_bytree': 0.9,  # hp.quniform('colsample_bytree', 0.5, 1.0, 0.01),
                'reg_alpha': 0.5,  # hp.quniform('reg_alpha', 0.0, 1.0, 0.01),
                'reg_lambda': 0.0,  # hp.quniform('reg_lambda', 0.0, 1.0, 0.01),
            }
        else:
            print("Wrong param_name in LgbParamSelect")

    def get_tuning_params(self):
        tuning_list = []
        for k, v in self.space_dict.items():
            if isinstance(v, hyperopt.pyll.base.Apply):
                tuning_list.append(k)
        return tuning_list


class LgbSkParamSelect():
    classifier_score = "roc_auc"
    rand_state = 20180401
    param_dict = dict()

    def __init__(self, space_name: str):
        if space_name == "default":
            self.name = space_name
            self.param_dict = {
                'num_leaves': range(10, 110),
                'max_depth': range(10, 50),
                'learning_rate': np.arange(0.01, 1, 0.01),  # np.arange(0.1, 3, 0.2),
                'n_estimators': range(50, 150, 10),  # range(100, 5000, 300),
                'min_split_gain': [0.0],  # np.arange(0.0, 1, 0.1),
                'min_child_weight': [0.01],  # np.arange(0.001, 1, 0.05),
                'min_child_samples': [1],  # range(20, 100, 10),
                'subsample': [0.9],  # np.arange(0.5, 1.001, 0.1),
                'subsample_freq': [10],  # range(0, 100, 10),  # frequency for bagging
                'colsample_bytree': [0.9],  # np.arange(0.5, 1.001, 0.1),
                'reg_alpha': [0.5],  # np.arange(0.0, 5.001, 0.5),
                'reg_lambda': [0.0],  # np.arange(0.0, 5.001, 0.5),
                'class_weight': 'balanced',
            }
        else:
            print("Wrong param_name in LgbParamSelect")


def get_dict_string(params_dict: dict):
    string = '{'
    for key in params_dict.keys():
        string += f'{key}: {params_dict[key]}, '
    string += '}'
    return string


g_eval_I = 0
g_HP_SCORE = 99999.9
g_BEST_ITER = 0
g_BEST_PARAM = ''
def hyperopt_tuning_model(lgb_model, param_select: LgbHpParamSelect, sample_df_, cv_, target_name):
    sample_X = sample_df_.drop(target_name, axis=1)
    sample_y = sample_df_[target_name]

    def objective(params):
        global g_eval_I, g_HP_SCORE, g_BEST_PARAM, g_BEST_ITER
        g_eval_I += 1
        with timer("Calc HP_score(cross_val_score)"):
            lgb_model.set_params(**params)
            score = cross_val_score(lgb_model, sample_X, sample_y, cv=cv_, scoring='roc_auc', n_jobs=1)
            dict_str = get_dict_string(params)
            hp_score = 1 - score.mean()
            if hp_score < g_HP_SCORE:
                g_HP_SCORE = hp_score
                g_BEST_ITER = g_eval_I
                g_BEST_PARAM = dict_str
            print(f'<{g_eval_I}>===hp_score={hp_score} | params={dict_str}')
            print(f'@@@ Until now best iter=<{g_BEST_ITER}>~~~~~g_HP_SCORE={g_HP_SCORE} | g_BEST_PARAM={g_BEST_PARAM}')
        return hp_score

    # The Trials object will store details of each iteration
    trials = Trials()

    # Run the hyperparameter search using the tpe algorithm
    best = fmin(objective,
                param_select.space_dict,
                algo=tpe.suggest,
                max_evals=5000,  # Allow up to this many function evaluations before returning.
                trials=trials,
                verbose=100)  # no real effect
    return best, trials


def show_hp_result(best_space_, trial_history_: Trials, hp_param_space, ouput_file: bool=False):
    best_params = space_eval(hp_param_space, best_space_)
    print("HP: best params=")
    pprint(best_params)
    Logger.info(f"使用HyperOpt调优得到最佳参数:\n{best_params}")

    result_cols = ['hp_score']
    param_names = list(trial_history_.vals.keys())
    param_names.sort()
    result_cols.extend(trial_history_.vals.keys())
    result_df = pd.DataFrame(columns=result_cols)
    for tid in trial_history_.tids:
        curr_param_space = dict()
        for param_name in param_names:
            curr_param_space[param_name] = trial_history_.vals[param_name][tid]
        curr_all_param = space_eval(hp_param_space, curr_param_space)
        for param_name in param_names:
            result_df.loc[tid, param_name] = curr_all_param[param_name]
        result_df.loc[tid, 'hp_score'] = trial_history_.results[tid]['loss']
    if ouput_file:
        def save_cv_result(file_):
            str_time = time.strftime("%m-%d_%H_%M", time.localtime(time.time()))
            base_dir = os.path.dirname(os.path.abspath(file_))
            csv_dir = base_dir + '/hp_result'
            if not os.path.exists(csv_dir):
                os.makedirs(csv_dir)
            return os.path.join(csv_dir, os.path.basename(file_).split('.py')[0] + f'_HP_result{str_time}.csv')
        result_df.sort_values(by=['hp_score'], ascending=True, inplace=True)
        result_df.to_csv(save_cv_result(__file__), index=False)
    Logger.info('HyperOpt Tuning Details: \n{}'.format(result_df))
    print(f'HP details: \n{result_df}')
    return best_params


def grid_search_tuning_model(lgb_model, param_select: LgbSkParamSelect, sample_df, cv_: list, target_name, search_switch):
    sample_X = sample_df.drop(target_name, axis=1)
    sample_y = sample_df[target_name]
    print(f'search_switch={search_switch}')

    if search_switch == "grid_search":
        clf = GridSearchCV(estimator=lgb_model,
                           param_grid=param_select.param_dict,
                           n_jobs=1,
                           cv=cv_,
                           scoring=param_select.classifier_score,
                           verbose=10,  # 2 print [CV] param and time, 10 add print score
                           refit=False)
    else:
        clf = RandomizedSearchCV(estimator=lgb_model,
                                 param_distributions=param_select.param_dict,
                                 n_iter=100,
                                 n_jobs=1,
                                 cv=cv_,
                                 scoring=param_select.classifier_score,
                                 verbose=10,  # 2 print [CV] param and time, 10 add print score
                                 refit=False)
    clf.fit(sample_X, sample_y)

    pprint(clf.best_params_)
    return clf


def use_best_pred_valid(best_params_, sample_df, cv_, target_name):
    sample_X = sample_df.drop(target_name, axis=1)
    sample_y = sample_df[target_name]

    Logger.info(f'使用最佳参数对每一个进行评估')
    for train_idx, valid_idx in cv_:
        train_X = sample_X.iloc[train_idx]
        train_y = sample_y.iloc[train_idx]
        valid_X = sample_X.iloc[valid_idx]
        valid_y = sample_y.iloc[valid_idx]
        best_model_ = lgb.LGBMClassifier(**best_params_)
        best_model_.fit(X=train_X, y=train_y)
        y_pred = best_model_.predict(valid_X)
        y_prob = best_model_.predict_proba(valid_X)[:, 1]
        auc_score = roc_auc_score(y_true=valid_y, y_score=y_prob)
        Logger.info(f"This fold validation dataset auc_score = {auc_score}")
        Logger.info(f'Precision, recall, f1-score:\n{classification_report(y_true=valid_y, y_pred=y_pred)}')


def get_best_param_fitted_model(best_model, sample_df, target_name):
    sample_X = sample_df.drop(target_name, axis=1)
    sample_y = sample_df[target_name]
    best_model.fit(X=sample_X, y=sample_y)
    return best_model


def show_feature_importance(lgb_model: lgb.LGBMClassifier, sample_df: pd.DataFrame, target_name: str):
    feats_name = list(sample_df.drop(target_name, axis=1).columns)
    importance = lgb_model.feature_importances_
    feats_ser = pd.Series(importance, index=feats_name)
    if platform.system() == 'Windows':
        feats_ser.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(17, 8))
        plt.title('LightGBM Feature Importance')
        plt.xlabel('relative importance')
        plt.show()


def print_param(cv_grid_params):
    Logger.info('选取的模型参数为：')
    Logger.info("param_name = '{}'".format(cv_grid_params.name))
    Logger.info("regression loss = {}".format(cv_grid_params.classifier_score))
    Logger.info("rand_state = {}".format(cv_grid_params.rand_state))
    Logger.info("param_dict = {")
    search_param_list = []
    for k, v in cv_grid_params.param_dict.items():
        Logger.info("\t'{}' = {}".format(k, v))
        if len(v) > 1:
            search_param_list.append(k)
    Logger.info("}")
    search_param_list.sort()
    return search_param_list


def get_cv_result_df(cv_results_, adjust_paras, n_cv):
    cols = ['mean_test_score', 'mean_train_score', 'mean_fit_time']
    for param_ in adjust_paras:
        cols.append('param_{}'.format(param_))
    for i in range(n_cv):
        cols.append('split{}_test_score'.format(i))
    for i in range(n_cv):
        cols.append('split{}_train_score'.format(i))
    return pd.DataFrame(data={key: cv_results_[key] for key in cols}, columns=cols)


def show_CV_result(search_clf, adjust_paras, classifi_scoring, output_file: bool):
    # pprint(reg.cv_results_)
    Logger.info('XXXXX查看CV的结果XXXXXX')
    Logger.info('{}: MAX of mean_test_score = {}'.format(classifi_scoring, search_clf.cv_results_.get('mean_test_score').max()))
    Logger.info('{}: MAX of mean_train_score = {}'.format(classifi_scoring, search_clf.cv_results_.get('mean_train_score').max()))
    n_splits = len(search_clf.cv) if isinstance(search_clf.cv, (list, tuple)) else search_clf.cv.n_splits
    cv_result_df = get_cv_result_df(search_clf.cv_results_, adjust_paras, n_splits)
    Logger.info('\n对各组调参参数的交叉训练验证细节为：\n{}'.format(cv_result_df))
    if output_file:
        def save_cv_result(file_):
            str_time = time.strftime("%m-%d_%H_%M", time.localtime(time.time()))
            base_dir = os.path.dirname(os.path.abspath(file_))
            csv_dir = base_dir + '/cv_result'
            if not os.path.exists(csv_dir):
                os.makedirs(csv_dir)
            return os.path.join(csv_dir, os.path.basename(file_).split('.py')[0] + f'_SK_result{str_time}.csv')
        cv_result_df.to_csv(save_cv_result(__file__), index=False)
    if len(adjust_paras) == 1 and platform.system() == 'Windows':
        every_para_score = pd.Series()
        every_para_score.name = adjust_paras[0]
        for i in range(len(search_clf.cv_results_.get('mean_test_score'))):
            record_param_value = search_clf.cv_results_.get('params')[i].get(adjust_paras[0])
            if isinstance(record_param_value, tuple):
                record_param_value = '{}'.format(reduce(lambda n_h, n_h1: str(n_h) + '_' + str(n_h1), record_param_value))
            every_para_score.loc[record_param_value] = search_clf.cv_results_.get('mean_test_score')[i]
        every_para_score.plot(kind='line', title=u'模型参数{}和评分{}的变化图示'.format(adjust_paras[0], classifi_scoring),
                              style='o-')
        plt.show()
    print('best_score_ = {}'.format(search_clf.best_score_))
    Logger.info('reg.best_score_: %f' % search_clf.best_score_)
    for param_name in sorted(search_clf.best_params_.keys()):
        if param_name in adjust_paras:
            Logger.info("调参选择为%s: %r" % (param_name, search_clf.best_params_[param_name]))


if __name__ == '__main__':
    # Get dataframe
    data_reader = DataReader(file_from='by_day__by_test_time', feats_construct='simplest')
    sample_df, cv_iterable, target_name = data_reader.get_train_feats_df("LGB")

    # Use GridSearch to coarse tuning and HyperOpt to fine tuning
    tuning_type = 'hp'  # 'sk' or 'hp'

    # Define model and tuning
    lgb_model = lgb.LGBMClassifier()
    if tuning_type == 'sk':
        print('Use sklearn GridSearch to tuning')
        param = LgbSkParamSelect("default")
        adjust_para_list = print_param(param)
        search_CV_model = grid_search_tuning_model(lgb_model, param, sample_df, cv_iterable, target_name,
                                                   search_switch="random_search")

        # See the CV result
        show_CV_result(search_CV_model, adjust_paras=adjust_para_list, classifi_scoring=param.classifier_score,
                       output_file=True)

        # Get best params model
        best_model = lgb.LGBMClassifier(**search_CV_model.best_params_)
    else:
        param = LgbHpParamSelect("default")
        hp_params = param.get_tuning_params()
        if len(hp_params) == 0:
            print('Just use dedicated params to predict')
            best_params = param.space_dict
            # Use best_params to predict validation data.
            use_best_pred_valid(best_params, sample_df, cv_iterable, target_name)
        else:
            print('Use HyperOpt GridSearch to tuning')
            best_space, trial_history = hyperopt_tuning_model(lgb_model, param, sample_df, cv_iterable, target_name)

            # See the HyperOpt result
            best_params = show_hp_result(best_space, trial_history, param.space_dict, ouput_file=True)

        # Get best params model
        best_model = lgb.LGBMClassifier(**best_params)

    # Use best params to fit on sample dataset get LGB model
    lgb_model = get_best_param_fitted_model(best_model, sample_df, target_name)

    # Show importance of features
    show_feature_importance(lgb_model, sample_df, target_name)



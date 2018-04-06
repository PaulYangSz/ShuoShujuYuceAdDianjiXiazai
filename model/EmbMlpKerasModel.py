"""
Use embedding and MLP to train and predict
"""
import platform
from functools import reduce

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import gc
import time
import os
from pprint import pprint
import hyperopt
from hyperopt import hp, fmin, tpe, Trials
from hyperopt import space_eval
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedKFold, train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score
import logging
import logging.config
from keras.layers import Input, Dropout, Dense, BatchNormalization, \
    Activation, concatenate, Embedding, Flatten
from keras.models import Model
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping  # , TensorBoard
from keras import backend as K
from keras.optimizers import Adam, Adadelta, Adagrad, Adamax, RMSprop
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted, check_X_y

from model.DataReader import DataReader, timer
from model.LgbModel import print_param, grid_search_tuning_model, show_CV_result, \
    hyperopt_tuning_model, show_hp_result, get_best_param_fitted_model

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


class EmbMlpHpParamSelect():
    classifier_score = "roc_auc"
    rand_state = 20180401
    space_dict = dict()

    def __init__(self, space_name: str):
        if space_name == "default":
            self.name = space_name
            self.space_dict = {
                'ip_dim': 50,  # 20 + hp.randint('ip_dim', 100),
                'app_dim': 20,  # 20 + hp.randint('app_dim', 100),
                'device_dim': 30,  # 20 + hp.randint('device_dim', 100),
                'os_dim': 20,  # 20 + hp.randint('os_dim', 100),
                'channel_dim': 20,  # 20 + hp.randint('channel_dim', 100),
                'click_time_dim': 10,  # 20 + hp.randint('click_time_dim', 100),
                'bn_flag': True,  # hp.choice('bn_flag', [True, False]),
                'dense_layers_unit': (128, 64),  # hp.choice('dense_layers_unit', [(128, 64)]),
                'drop_out': (0.2, 0.2),  # hp.choice('drop_out', [(0.2, 0.2)]),
                'active': ('relu', 'relu'),  # hp.choice('active', [('relu', 'relu')]),
                'epochs': 2,  # hp.choice('epochs', [1, 2, 3]),
                'batch_size': 512 * 3,  # hp.quniform('reg_lambda', 0.0, 1.0, 0.01),
                'lr_init': 0.015,  # hp.quniform('lr_init', 0.01, 0.04, 0.001),
                'lr_final': 0.007,  # hp.quniform('lr_final', 0.001, 0.01, 0.001),
            }
        else:
            print("Wrong param_name in LgbParamSelect")

    def get_tuning_params(self):
        tuning_list = []
        for k, v in self.space_dict.items():
            if isinstance(v, hyperopt.pyll.base.Apply):
                tuning_list.append(k)
        return tuning_list


class EmbMlpSkParamSelect():
    classifier_score = "roc_auc"
    rand_state = 20180401
    param_dict = dict()

    def __init__(self, space_name: str):
        if space_name == "default":
            self.name = space_name
            self.param_dict = {
                'ip_dim': [50],  # 20 + hp.randint('ip_dim', 100),
                'app_dim': [20],  # 20 + hp.randint('app_dim', 100),
                'device_dim': [30],  # 20 + hp.randint('device_dim', 100),
                'os_dim': [20],  # 20 + hp.randint('os_dim', 100),
                'channel_dim': [20],  # 20 + hp.randint('channel_dim', 100),
                'click_time_dim': [10],  # 20 + hp.randint('click_time_dim', 100),
                'bn_flag': [True],
                'dense_layers_unit': [(128, 64)],  # hp.choice('dense_layers_unit', [(128, 64)]),
                'drop_out': [(0.2, 0.2)],  # hp.choice('drop_out', [(0.2, 0.2)]),
                'active': [('relu', 'relu')],  # hp.choice('active', [('relu', 'relu')]),
                'epochs': [4],  # hp.choice('epochs', [1, 2, 3]),
                'batch_size': [512*12],  # hp.quniform('reg_lambda', 0.0, 1.0, 0.01),
                'lr_init': [0.015],  # hp.quniform('lr_init', 0.01, 0.04, 0.001),
                'lr_final': [0.007],  # hp.quniform('lr_final', 0.001, 0.01, 0.001),
            }
        else:
            print("Wrong param_name in LgbParamSelect")


MAX_IP = MAX_APP = MAX_DEVICE = MAX_OS = MAX_CHANNEL = MAX_CLICK_TIME = -1
MAX_DEVICE_OS_N = MAX_APP_CH_N = MAX_DEVICE_CH_N = MAX_OS_CH_N = MAX_CH_OS_N = -1
FUNC_GET_KERAS_INPUT = None


class EmbMlpModel(BaseEstimator, ClassifierMixin):
    """ An sklearn-API regressor.
    Model: Embedding all category -> Concat[] ->  Dense -> Output
    Parameters
    ----------
    demo_param : All tuning parameters should be set in __init__()
        A parameter used for demonstation of how to pass and store paramters.
    Attributes
    ----------
    X_ : array, shape = [n_samples, n_features]
        The input passed during :meth:`fit`
    y_ : array, shape = [n_samples]
        The labels passed during :meth:`fit`
    """

    def __init__(self, ip_dim=50, app_dim=20, device_dim=30, os_dim=20, channel_dim=20, click_time_dim=10,
                 device_os_n_dim=10, app_ch_n_dim=10, device_ch_n_dim=10, os_ch_n_dim=10, ch_os_n_dim=10,
                 bn_flag=True, dense_layers_unit=(128, 64), drop_out=(0.2, 0.2), active=('relu', 'relu'),
                 epochs=1, batch_size=512*3, lr_init=0.015, lr_final=0.007):
        self.ip_dim = ip_dim
        self.app_dim = app_dim
        self.device_dim = device_dim
        self.os_dim = os_dim
        self.channel_dim = channel_dim
        self.click_time_dim = click_time_dim
        self.device_os_n_dim = device_os_n_dim
        self.app_ch_n_dim = app_ch_n_dim
        self.device_ch_n_dim = device_ch_n_dim
        self.os_ch_n_dim = os_ch_n_dim
        self.ch_os_n_dim = ch_os_n_dim
        self.bn_flag = bn_flag
        self.dense_layers_unit = dense_layers_unit
        self.drop_out = drop_out
        self.active = active
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr_init = lr_init
        self.lr_final = lr_final

        self.mlp_model = self.get_embedding_mlp_model()

    def __del__(self):
        print('%%%%%%%%__del__')
        if K.backend() == 'tensorflow':
            K.clear_session()

    def get_embedding_mlp_model(self):
        # Inputs
        # ip = Input(shape=[1], name="ip")
        app = Input(shape=[1], name='app')
        device = Input(shape=[1], name='device')
        os = Input(shape=[1], name='os')
        channel = Input(shape=[1], name='channel')
        click_time = Input(shape=[1], name='click_time')
        device_os_n = Input(shape=[1], name='device_os_n')
        app_ch_n = Input(shape=[1], name='app_ch_n')
        device_ch_n = Input(shape=[1], name='device_ch_n')
        os_ch_n = Input(shape=[1], name='os_ch_n')
        ch_os_n = Input(shape=[1], name='ch_os_n')

        # Embedding all category input to vectors
        # each int value must in [0, max_int)
        # emb_ip = Embedding(input_dim=MAX_IP, output_dim=self.ip_dim)(ip)
        emb_app = Embedding(MAX_APP, self.app_dim)(app)  # [None, STEPS, emb_size]
        emb_device = Embedding(MAX_DEVICE, self.device_dim)(device)
        emb_os = Embedding(MAX_OS, self.os_dim)(os)
        emb_channel = Embedding(MAX_CHANNEL, self.channel_dim)(channel)
        emb_click_time = Embedding(MAX_CLICK_TIME, self.click_time_dim)(click_time)
        emb_device_os_n = Embedding(MAX_DEVICE_OS_N, self.device_os_n_dim)(device_os_n)
        emb_app_ch_n = Embedding(MAX_APP_CH_N, self.app_ch_n_dim)(app_ch_n)
        emb_device_ch_n = Embedding(MAX_DEVICE_CH_N, self.device_ch_n_dim)(device_ch_n)
        emb_os_ch_n = Embedding(MAX_OS_CH_N, self.os_ch_n_dim)(os_ch_n)
        emb_ch_os_n = Embedding(MAX_CH_OS_N, self.ch_os_n_dim)(ch_os_n)

        # concatenate to main layer
        main_layer = concatenate([#Flatten()(emb_ip),
                                  Flatten()(emb_app),  # [None, STEPS, dim] -> [None, STEPS * dim]
                                  Flatten()(emb_device),
                                  Flatten()(emb_os),
                                  Flatten()(emb_channel),
                                  Flatten()(emb_click_time),
                                  Flatten()(emb_device_os_n),
                                  Flatten()(emb_app_ch_n),
                                  Flatten()(emb_device_ch_n),
                                  Flatten()(emb_os_ch_n),
                                  Flatten()(emb_ch_os_n)])

        # MLP
        for i in range(len(self.dense_layers_unit)):
            main_layer = Dense(units=self.dense_layers_unit[i])(main_layer)
            main_layer = Dropout(self.drop_out[i])(main_layer)
            if self.bn_flag:
                main_layer = BatchNormalization()(main_layer)
            main_layer = Activation(activation=self.active[i])(main_layer)

        # output
        output = Dense(1, activation='sigmoid')(main_layer)

        # Model
        model = Model(inputs=[app, device, os, channel, click_time, device_os_n, app_ch_n, device_ch_n, os_ch_n, ch_os_n],
                      outputs=output)

        # optimizer
        optimizer = Adam(lr=0.001, decay=0.0)

        model.compile(loss="binary_crossentropy", optimizer=optimizer)  # , metrics=
        return model

    def fit(self, X, y):
        # Check that X and y have correct shape
        # X, y = check_X_y(X, y)  # It will change type(X) to np.ndarray
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.X_ = X
        self.y_ = y

        # FITTING THE MODEL
        steps = int(X.shape[0] / self.batch_size) * self.epochs
        # final_lr=init_lr * (1/(1+decay))**(steps-1)
        exp_decay = lambda init, final, step_num: (init / final) ** (1 / (step_num - 1)) - 1
        lr_decay = exp_decay(self.lr_init, self.lr_final, steps)
        log_subdir = '_'.join(['ep', str(self.epochs),
                               'bs', str(self.batch_size),
                               'lrI', str(self.lr_init),
                               'lrF', str(self.lr_final)])
        K.set_value(self.mlp_model.optimizer.lr, self.lr_init)
        K.set_value(self.mlp_model.optimizer.decay, lr_decay)

        keras_X = FUNC_GET_KERAS_INPUT(X)
        keras_fit_start = time.time()
        history = self.mlp_model.fit(keras_X, y, epochs=self.epochs, batch_size=self.batch_size,
                                     validation_split=0.,  # 0.01
                                     # callbacks=[TensorBoard('./logs/'+log_subdir)],
                                     verbose=1)  # 0 = 安静模式, 1 = 进度条, 2 = 每轮一行
        Logger.info('[self.emb_GRU_model.fit] cost {:.4f}s'.format(time.time() - keras_fit_start))
        print('[self.emb_GRU_model.fit] cost {:.4f}s'.format(time.time() - keras_fit_start))

        # Return the regressor
        return self

    def predict_proba(self, X):
        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        keras_X = FUNC_GET_KERAS_INPUT(X)
        emb_mlp_y_proba = np.empty(shape=[X.shape[0], 2])
        proba_ = self.mlp_model.predict(keras_X, batch_size=70000, verbose=10)
        proba_ = proba_.reshape(proba_.shape[0])
        emb_mlp_y_proba[:, 1] = proba_
        emb_mlp_y_proba[:, 0] = 1 - emb_mlp_y_proba[:, 1]
        return emb_mlp_y_proba

    def predict(self, X):
        emb_mlp_y = self.predict_proba(X)[:, 1]
        emb_mlp_y = (emb_mlp_y > 0.499999).astype(np.int8)
        return emb_mlp_y


def use_best_pred_valid(best_params_, sample_df, cv_, target_name):
    sample_X = sample_df.drop(target_name, axis=1)
    sample_y = sample_df[target_name]

    Logger.info(f'使用最佳参数对每一个进行评估')
    for train_idx, valid_idx in cv_:
        train_X = sample_X.iloc[train_idx]
        train_y = sample_y.iloc[train_idx]
        valid_X = sample_X.iloc[valid_idx]
        valid_y = sample_y.iloc[valid_idx]
        best_model_ = EmbMlpModel(**best_params_)
        best_model_.fit(X=train_X, y=train_y)
        y_prob = best_model_.predict_proba(valid_X)[:, 1]
        y_pred = (y_prob > 0.499999).astype(np.int8)
        auc_score = roc_auc_score(y_true=valid_y, y_score=y_prob)
        Logger.info(f"This fold validation dataset auc_score = {auc_score}")
        Logger.info(f'Precision, recall, f1-score:\n{classification_report(y_true=valid_y, y_pred=y_pred)}')
        del best_model_
        gc.collect()


def save_test_result(fitted_model, test_df, file_name):
    sub_df = pd.DataFrame()
    sub_df['click_id'] = test_df['click_id']
    sub_df['is_attributed'] = fitted_model.predict_proba(test_df)[:, 1]
    sub_df.to_csv(file_name, index=False)


if __name__ == "__main__":
    # Get dataframe
    data_reader = DataReader(file_from='by_day__by_test_time', feats_construct='add_day_stat', verify_code=False)
    sample_df, cv_iterable, target_name = data_reader.get_train_feats_df("MLP")
    test_df = data_reader.get_test_feats_df("MLP")

    # Get model constant params
    MAX_IP = data_reader.max_ip
    MAX_APP = data_reader.max_app
    MAX_DEVICE = data_reader.max_device
    MAX_OS = data_reader.max_os
    MAX_CHANNEL = data_reader.max_channel
    MAX_CLICK_TIME = data_reader.max_click_time
    MAX_DEVICE_OS_N = data_reader.max_device_os_n
    MAX_APP_CH_N = data_reader.max_app_ch_n
    MAX_DEVICE_CH_N = data_reader.max_device_ch_n
    MAX_OS_CH_N = data_reader.max_os_ch_n
    MAX_CH_OS_N = data_reader.max_ch_os_n
    FUNC_GET_KERAS_INPUT = data_reader.get_keras_input

    # Use GridSearch to coarse tuning and HyperOpt to fine tuning
    tuning_type = 'sk'  # 'sk' or 'hp'

    # Define model and tuning
    if tuning_type == 'sk':
        print('~Use sklearn GridSearch to tuning')
        param = EmbMlpSkParamSelect("default")
        adjust_para_list = print_param(param)
        emb_mlp_model = EmbMlpModel()
        emb_mlp_model.mlp_model.summary()
        search_CV_model = grid_search_tuning_model(emb_mlp_model, param, sample_df, cv_iterable, target_name,
                                                   search_switch="grid_search")  # random_search

        # See the CV result
        show_CV_result(search_CV_model, adjust_paras=adjust_para_list, classifi_scoring=param.classifier_score,
                       output_file=True)

        # Get best params model
        best_model = EmbMlpModel(**search_CV_model.best_params_)
    else:
        param = EmbMlpHpParamSelect("default")
        hp_params = param.get_tuning_params()
        if len(hp_params) == 0:
            print('~Just use dedicated params to predict')
            best_params = param.space_dict
            # Use best_params to predict validation data.
            use_best_pred_valid(best_params, sample_df, cv_iterable, target_name)
        else:
            print('~Use HyperOpt GridSearch to tuning')
            emb_mlp_model = EmbMlpModel()
            emb_mlp_model.mlp_model.summary()
            best_space, trial_history = hyperopt_tuning_model(emb_mlp_model, param, sample_df, cv_iterable, target_name)

            # See the HyperOpt result
            best_params = show_hp_result(best_space, trial_history, param.space_dict, ouput_file=True)

        # Get best params model
        best_model = EmbMlpModel(**best_params)

    # Use best params to fit on sample dataset get LGB model
    fitted_model = get_best_param_fitted_model(best_model, sample_df, target_name)

    # Generate the test_result
    file_name = "../output/emb_mlp_sub({}).csv".format(time.strftime("%Y.%m.%d-%H%M"))
    save_test_result(fitted_model, test_df, file_name)



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
from sklearn.base import BaseEstimator, RegressorMixin
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


class SelfLocalRegressor(BaseEstimator, RegressorMixin):
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
                 bn_flag=True, dense_layers_unit=(128, 64), drop_out=(),
                 epochs=1, batch_size=512*3, lr_init=0.015, lr_final=0.007):
        self.ip_dim = ip_dim
        self.app_dim = app_dim
        self.device_dim = device_dim
        self.os_dim = os_dim
        self.channel_dim = channel_dim
        self.click_time_dim = click_time_dim
        self.bn_flag = bn_flag
        self.dense_layers_unit = dense_layers_unit
        self.drop_out = drop_out
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr_init = lr_init
        self.lr_final = lr_final

    def __del__(self):
        print('%%%%%%%%__del__')
        if K.backend() == 'tensorflow':
            K.clear_session()

    def get_embedding_mlp_model(self):
        # Inputs
        ip = Input(shape=[1], name="ip")
        app = Input(shape=[1], name='app')
        device = Input(shape=[1], name='device')
        os = Input(shape=[1], name='os')
        channel = Input(shape=[1], name='channel')
        click_time = Input(shape=[1], name='click_time')

        # Embedding all category input to vectors


if __name__ == "__main__":
    # Get dataframe
    data_reader = DataReader(file_from='by_day__by_test_time', feats_construct='simplest')
    sample_df, cv_iterable, target_name = data_reader.get_train_feats_df()


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
from keras import initializers, regularizers, constraints
from keras.engine.topology import Layer, InputSpec
from sklearn.preprocessing import LabelEncoder
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
                'dense_layers_unit': [(256, 128)],  # hp.choice('dense_layers_unit', [(128, 64)]),
                'drop_out': [(0.2, 0.2)],  # hp.choice('drop_out', [(0.2, 0.2)]),
                'active': [('relu', 'relu')],  # hp.choice('active', [('relu', 'relu')]),
                'epochs': [4],  # hp.choice('epochs', [1, 2, 3]),
                'batch_size': [512*12],  # hp.quniform('reg_lambda', 0.0, 1.0, 0.01),
                'lr_init': [0.015],  # hp.quniform('lr_init', 0.01, 0.04, 0.001),
                'lr_final': [0.001],  # hp.quniform('lr_final', 0.001, 0.01, 0.001),
            }
        else:
            print("Wrong param_name in LgbParamSelect")

    def get_model_param(self):
        model_params = {}
        for k, v in self.param_dict.items():
            model_params[k] = v[0]
        return model_params


class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        # self.init = initializations.get('glorot_uniform')
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        # W的shape为(input的最后一维长度相等,) 因为不是Seq2Seq，而是Seq2One，所以W的作用相当于y的前一个状态值，用于和每个Xi计算相似度
        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        # feats_dim为input的最后一维长度
        self.features_dim = input_shape[-1]

        if self.bias:
            # b与input第一维数据长度相等
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        input_shape = K.int_shape(x)  # input的维数==3
        features_dim = self.features_dim  # input最后一维的长度
        step_dim = input_shape[1]  # input中间维度的长度也就是step的长度
        # eij = K.reshape([batch_size*steps, feats_dim] dot* [feats_dim, 1], (-1, steps)) = [batch_size, steps]
        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))
        if self.bias:
            eij += self.b[:input_shape[1]]
        eij = K.tanh(eij)  # value to [-1, 1]
        a = K.exp(eij)
        if mask is not None:
            a *= K.cast(mask, K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())  # exp() then sum() is Softmax
        a = K.expand_dims(a)  # [batch_size, steps, 1]
        weighted_input = x * a  # [batch_size, steps, feats_dim] 直接用x与系数a相乘
        return K.sum(weighted_input, axis=1)

    def get_w(self, x, mask=None):
        input_shape = K.int_shape(x)
        features_dim = self.features_dim
        step_dim = input_shape[1]
        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))
        if self.bias:
            eij += self.b[:input_shape[1]]
        eij = K.tanh(eij)
        a = K.exp(eij)
        if mask is not None:
            a *= K.cast(mask, K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())  # [batch_size, steps]
        return a

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.features_dim


MAX_FEATS_VALUES = dict()  # Control the emb_feat's max value(input_dim)
ALL_FEAT_COLS = list()  # Control the model's Input
EMB_FEATS_DIMS = dict()  # Control the emb_feat's output_dim, for now use get_dim_from_max() instead.
FUNC_GET_KERAS_INPUT = None


def get_dim_from_max(max_value):
    if max_value < 30:
        return 5
    elif max_value < 120:
        return 7
    elif max_value < 1000:
        return 10
    else:
        return 20


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
                 iptime_app_n_dim=10, iptime_device_n_dim=10, iptime_os_n_dim=10, iptime_ch_n_dim=10, iptime_click_n_dim=20,
                 bn_flag=True, dense_layers_unit=(128, 64), drop_out=(0.2, 0.2), active=('relu', 'relu'),
                 epochs=1, batch_size=512*3, lr_init=0.015, lr_final=0.007):
        self.ip_dim = ip_dim
        self.app_dim = app_dim
        self.device_dim = device_dim
        self.os_dim = os_dim
        self.channel_dim = channel_dim
        self.click_time_dim = click_time_dim
        self.iptime_app_n_dim = iptime_app_n_dim
        self.iptime_device_n_dim = iptime_device_n_dim
        self.iptime_os_n_dim = iptime_os_n_dim
        self.iptime_ch_n_dim = iptime_ch_n_dim
        self.iptime_click_n_dim = iptime_click_n_dim
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
        Logger.info(f"^^^^^When construct model, ALL_FEAT_COLS = \n{ALL_FEAT_COLS}")
        # Inputs
        input_dict = {}
        for col in ALL_FEAT_COLS:
            input_dict[col] = Input(shape=[1], name=col)

        # Embedding all category input to vectors
        # each int value must in [0, max_int)
        emb_dict = {}
        for col in ALL_FEAT_COLS:
            if not (col.endswith('_var') or col.endswith('_ent') or col.endswith('_rate')):
                dim = get_dim_from_max(MAX_FEATS_VALUES[col])
                emb_dict[col] = Embedding(MAX_FEATS_VALUES[col], dim)(input_dict[col])

        # concatenate to main layer
        concat_list = []
        for col in emb_dict:
            concat_list.append(Flatten()(emb_dict[col]))  # [None, STEPS, dim] -> [None, STEPS * dim]
        for col in ALL_FEAT_COLS:
            if col.endswith('_var') or col.endswith('_ent') or col.endswith('_rate'):
                concat_list.append(input_dict[col])
        main_layer = concatenate(concat_list)

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
        model = Model(inputs=list(input_dict.values()),
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
        print(keras_X)
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
    sub_df['click_id'] = test_df['click_id']  # .astype(np.int32)
    sub_df['is_attributed'] = fitted_model.predict_proba(test_df)[:, 1]
    sub_df.to_csv(file_name, index=False)  # , float_format='%.8f'


def convert_feats_int_type(feat_ser):
    feat_max = feat_ser.max()
    if feat_max < np.iinfo('uint8').max:
        return feat_ser.astype('uint8')
    elif feat_max < np.iinfo('uint16').max:
        return feat_ser.astype('uint16')
    elif feat_max < np.iinfo('uint32').max:
        return feat_ser.astype('uint32')
    else:
        return feat_ser.astype('uint64')


def convert_feats_float_type(feat_ser):
    feat_max = feat_ser.max()
    if feat_max < np.finfo('float16').max:
        return feat_ser.astype('float16')
    elif feat_max < np.finfo('float32').max:
        return feat_ser.astype('float32')
    else:
        return feat_ser.astype('float64')


def label_feats_and_set_max(sample_df_: pd.DataFrame, test_df_: pd.DataFrame, le_cols):
    len_sample = len(sample_df_)
    all_df: pd.DataFrame = sample_df_.append(test_df_)
    for col in all_df.columns:
        if col not in ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed', 'click_id']:
            if np.issubdtype(all_df[col].dtype, np.floating):
                all_df[col] = convert_feats_float_type(all_df[col])
            elif np.issubdtype(all_df[col].dtype, np.integer):
                all_df[col] = convert_feats_int_type(all_df[col])
            else:
                pass
    print(f"sample_df_.cols=\n{sample_df_.columns}, \ntest_df_.cols=\n{test_df_.columns}, \nall_df.cols=\n{all_df.columns}")
    print(f"@@@all_df.dtypes=\n{all_df.dtypes}")
    le = LabelEncoder()
    for lb_col in le_cols:
        all_df[lb_col] = le.fit_transform(all_df[lb_col])
        all_df[lb_col] = convert_feats_int_type(all_df[lb_col])
    del le
    gc.collect()
    print(f"@@@@@@@@@@after LabelEncoder all_df.dtypes=\n{all_df.dtypes}")
    global MAX_FEATS_VALUES, ALL_FEAT_COLS
    for col in all_df.columns:
        if col not in ['ip', 'is_attributed', 'click_id']:
            ALL_FEAT_COLS.append(col)
            MAX_FEATS_VALUES[col] = all_df[col].max() + 1
    Logger.info(f"将{str(le_cols)}转换为Label后，各特征取值上限为: \n{MAX_FEATS_VALUES}")
    _test_df = all_df[len_sample:].copy()
    _test_df['click_id'] = _test_df['click_id'].astype(np.int32)
    _sample_df = all_df[:len_sample].copy()
    _sample_df['is_attributed'] = _sample_df['is_attributed'].astype(np.uint8)
    del all_df
    gc.collect()
    return _sample_df, _test_df


def try_add_one_feat(sample_df_, cv_iterable_, target_name_, new_col, add_flag):
    if add_flag:
        if new_col != '':
            ALL_FEAT_COLS.append(new_col)
        Logger.info(f"+++++++++++++++++Try add new col<{new_col}> in ALL_FEAT_COLS")
    else:
        if new_col != '':
            ALL_FEAT_COLS.remove(new_col)
        Logger.info(f"-----------------Try remove new col<{new_col}> out of ALL_FEAT_COLS")
    param = EmbMlpSkParamSelect("default")
    model_ = EmbMlpModel(**param.get_model_param())
    model_.mlp_model.summary()
    sample_X = sample_df_.drop(target_name_, axis=1)
    sample_y = sample_df_[target_name_]
    train_X = sample_X.iloc[cv_iterable_[2][0]]
    train_y = sample_y.iloc[cv_iterable_[2][0]]
    test_X = sample_X.iloc[cv_iterable_[2][1]]
    test_y = sample_y.iloc[cv_iterable_[2][1]]
    model_.fit(train_X, train_y)
    y_prob = model_.predict_proba(test_X)[:, 1]
    auc_score = roc_auc_score(y_true=test_y, y_score=y_prob)
    Logger.info(f"^^^^^ after add/remove <{new_col}>, auc_score={auc_score}")
    if new_col != '':
        if add_flag:
            ALL_FEAT_COLS.remove(new_col)  # After add need remove for next new_col in
        else:
            ALL_FEAT_COLS.append(new_col)  # After remove need add for next new_col out
    del model_
    gc.collect()


if __name__ == "__main__":
    # Get dataframe
    data_reader = DataReader(file_from='by_day__by_test_time', feats_construct='add_time_interval_stat', time_interval='test_1hour', verify_code=False)
    sample_df, cv_iterable, target_name = data_reader.get_train_feats_df("MLP")
    test_df = data_reader.get_test_feats_df("MLP")
    # with timer(f"Add attributed feats to train and test df<attributed rate>"):
    #     sample_df, test_df = data_reader.add_attributed_stat_way(sample_df, test_df)

    # Continue to preprocess data
    need_label_cols = []
    for col in sample_df.columns:
        if col.endswith('_click_n'):
            need_label_cols.append(col)
    with timer("Use LabelEncoder().fit_transform to continue process data"):
        sample_df, test_df = label_feats_and_set_max(sample_df, test_df, need_label_cols)

    # Get model constant params
    FUNC_GET_KERAS_INPUT = data_reader.get_keras_input

    only_submit = False
    if not only_submit:
        try_add_each_feat = False
        if try_add_each_feat:
            try_add_flag = True
            attempt_cols = ['', 'ipapp_ch_n', 'ipapp_click_n', 'ipappos_ch_n', 'ipappos_click_n',
                            'ipappdevice_ch_n', 'ipappdevice_click_n']
            # attempt_cols = [''] + ALL_FEAT_COLS
            if try_add_flag:
                for col in attempt_cols[1:]:
                    try:
                        ALL_FEAT_COLS.remove(col)
                    except:
                        print(ALL_FEAT_COLS)
                        print(col)
                        assert False
            for alter_col in attempt_cols:  # '' means not add or remove
                try_add_one_feat(sample_df, cv_iterable, target_name, alter_col, try_add_flag)
        else:
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
    else:
        param = EmbMlpSkParamSelect("default")
        model_init = EmbMlpModel(**param.get_model_param())
        model_init.mlp_model.summary()
        # Use best params to fit on sample dataset get LGB model
        # sample_df = sample_df.iloc[cv_iterable[2][1]]  # 0,0: 8&9; 0,1: 7; 1,1: 8; 2,1: 9
        fitted_model = get_best_param_fitted_model(model_init, sample_df, target_name)

        # Generate the test_result
        file_name = "../output/emb_mlp_sub({}).csv".format(time.strftime("%Y.%m.%d-%H%M"))
        save_test_result(fitted_model, test_df, file_name)



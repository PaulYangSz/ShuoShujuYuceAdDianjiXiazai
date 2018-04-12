"""
DataReader is a class which is used to read dataset and return train/validation/test dataframe.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import gc
from contextlib import contextmanager
import time


import logging
import logging.config


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


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.3f} s')


DTYPES = {
    'ip': 'uint32',
    'app': 'uint16',
    'device': 'uint16',
    'os': 'uint16',
    'channel': 'uint16',
    'is_attributed': 'uint8',
    'click_id': 'uint32',
}


def datetime2cate(conver_by: str, _time: datetime):
    """
    Test time: 04:00——06:00,09:00——11:00,13:00——15:00
    :param conver_by: convert method
    :param _time: click datetime
    :return: int of category
    """
    _h = _time.hour
    _m = _time.minute
    _s = _time.second
    if conver_by == 'test_30mins':
        _half = 'upper' if _m < 30 else 'bottom'
        date2cate_map = {(4, 'upper'): 1, (4, 'bottom'): 2, (5, 'upper'): 3, (5, 'bottom'): 4, (6, 'upper'): 4,
                         (9, 'upper'): 5, (9, 'bottom'): 6, (10, 'upper'): 7, (10, 'bottom'): 8, (11, 'upper'): 8,
                         (13, 'upper'): 9, (13, 'bottom'): 10, (14, 'upper'): 11, (14, 'bottom'): 12, (15, 'upper'): 12}
        return date2cate_map[(_h, _half)]
    elif conver_by == 'test_1hour':
        date2cate_map = {4: 1, 5: 2, 6: 2,
                         9: 3, 10: 4, 11: 4,
                         13: 5, 14: 6, 15: 6}
        return date2cate_map[_h]
    else:
        return None


def update_max(max_old, data_df, col_str):
    return data_df[col_str].max() + 1 if data_df[col_str].max() > max_old else max_old + 1


def get_gp_from_dict(data_df, spec):
    Logger.info(f"GroupBy构造新特征<{spec['new']}>: by={spec['groupby']}, select={spec['select']}, agg={spec['agg_name']}")
    all_cols = spec['groupby'] + [spec['select']]
    if 'astype' in spec:
        gp = data_df[all_cols].groupby(by=spec['groupby'])[spec['select']].agg(spec['agg']).reset_index().rename(index=str, columns={spec['select']: spec['new']}).astype(spec['astype'])
    else:
        gp = data_df[all_cols].groupby(by=spec['groupby'])[spec['select']].agg(spec['agg']).reset_index().rename(index=str, columns={spec['select']: spec['new']})
    return gp


def get_value_counts_entroy(value_ser):
    proba = value_ser.value_counts().values
    if proba.size == 1:
        return 0
    else:
        proba = proba / proba.sum()
        return -np.dot(proba, np.log(proba))


# Aggregation function
def rate_calculation(x_attributed):
    """Calculate the attributed rate. Scale by confidence"""
    log_group = 100000  # 1000 views -> 60% confidence, 100 views -> 40% confidence
    rate = x_attributed.sum() / float(x_attributed.count())
    conf = np.min([1, np.log(x_attributed.count()) / log_group])
    return rate * conf


class DataReader:

    def __init__(self, file_from:str, feats_construct:str, time_interval:str, verify_code: bool):
        self.file_from = file_from
        self.feats_construct = feats_construct
        self.time_interval = time_interval
        self.n_rows = 100000 if verify_code else None
        # For one train_df will use StratifiedKFold to split train and validation
        # For train_df list each element will be loop as train and validation
        self.train_df_list = []
        self.test_df = None
        self.int_cols = ['ip', 'app', 'device', 'os', 'channel', 'is_attributed']
        self.target = 'is_attributed'
        self.time_cols = ['click_time', 'attributed_time']
        self.simplest_bool = False
        self.day_stat_bool = False
        self.max_ip = -1
        self.max_app = -1
        self.max_device = -1
        self.max_os = -1
        self.max_channel = -1
        self.max_click_time = -1

    def load_train(self):
        if self.file_from == 'by_day__by_test_time':
            for day in range(7, 10):
                day_path = '../data/day_with_test_hour/201711{:02d}_test_hour.csv'.format(day)
                day_df = pd.read_csv(day_path, dtype=DTYPES, nrows=self.n_rows)
                self.train_df_list.append(day_df)
        else:
            print(f"!!! Wrong param['file_from'] = '{self.file_from}'")

    def load_test(self):
        self.test_df = pd.read_csv('../data/test.csv', dtype=DTYPES, nrows=self.n_rows)

    def construct_feats(self, data_df: pd.DataFrame, model_name: str):
        if self.feats_construct == 'simplest':
            return self.simplest_way(data_df, model_name)
        elif self.feats_construct == "add_day_stat":
            return self.add_day_stat_way(data_df, model_name)
        elif self.feats_construct == "add_time_interval_stat":
            return self.add_time_interval_stat_way(data_df, model_name)
        else:
            print(f"!!! Wrong param['feats_construct'] = '{self.feats_construct}'")

    def get_train_feats_df(self, model_name: str):
        with timer("Loading train csv files"):
            self.load_train()
        train_feat_df = pd.DataFrame()
        each_len = []
        for train_df in self.train_df_list:
            with timer(f"Construct train feats df<'{self.feats_construct}'>"):
                each_len.append(train_df.shape[0])
                train_feat_df = pd.concat([train_feat_df, self.construct_feats(train_df, model_name)], axis=0, ignore_index=True)
                del train_df
                gc.collect()
        if model_name in ["MLP"]:
            self.make_emb_max(train_feat_df)
        self.train_df_list = []
        cv_index_list = []  # [(train_idx, test_idx), (train_idx, test_idx), ...]
        index_start = 0
        for len_ in each_len:
            train_idx = np.array(train_feat_df.index[0: index_start].tolist() + train_feat_df.index[index_start+len_:].tolist())
            test_idx = np.array(train_feat_df.index[index_start: index_start + len_].tolist())
            cv_index_list.append((train_idx, test_idx))
            index_start += len_
        return train_feat_df, cv_index_list, self.target

    def get_test_feats_df(self, model_name: str):
        with timer("Loading test csv file"):
            self.load_test()
        with timer(f"Construct test feats df<'{self.feats_construct}'>"):
            test_df = self.construct_feats(self.test_df, model_name)
            if model_name in ["MLP"]:
                self.get_emb_max(test_df)
            del self.test_df
            gc.collect()
        return test_df

    def get_keras_input(self, dataframe):
        X = dict()
        for col in dataframe.columns:
            if col not in ['ip', 'is_attributed', 'click_id']:
                X[col] = np.array(dataframe[col])
        return X

    def simplest_way(self, data_df, model_name):
        """
        对于LGB产生全部为cate类型的ip,app,device,os,channel,click_time
        对于MLP产生为int类型的ip,app,device,os,channel,click_time
        :param data_df:
        :param model_name:
        :return:
        """
        for col in self.int_cols:
            if col in data_df.columns:
                if model_name in ["LGB"]:
                    data_df[col] = data_df[col].astype('category')
                else:
                    pass
        data_df['click_time'] = pd.to_datetime(data_df['click_time'])
        data_df['click_time'] = data_df['click_time'].map(lambda t: datetime2cate(self.time_interval, t)).astype('uint8')
        if model_name in ["LGB"]:
            data_df['click_time'] = data_df['click_time'].astype('category')
        cols = self.int_cols[:5] + ['click_time', 'is_attributed'] if self.target in data_df.columns \
            else self.int_cols[:5] + ['click_time', 'click_id']
        self.simplest_bool = True
        return data_df[cols]

    def add_day_stat_way(self, data_df, model_name):
        data_df = self.simplest_way(data_df, model_name)
        # reset_index() can cast Series to Dataframe
        group_by_list = [
            {'groupby': ['device'], 'select': 'os', 'agg': pd.Series.nunique, 'agg_name': 'nunique', 'new': 'device_os_n', 'astype': np.int16},
            {'groupby': ['app'], 'select': 'channel', 'agg': pd.Series.nunique, 'agg_name': 'nunique', 'new': 'app_ch_n', 'astype': np.int16},
            {'groupby': ['device'], 'select': 'channel', 'agg': pd.Series.nunique, 'agg_name': 'nunique', 'new': 'device_ch_n', 'astype': np.int16},
            {'groupby': ['os'], 'select': 'channel', 'agg': pd.Series.nunique, 'agg_name': 'nunique', 'new': 'os_ch_n', 'astype': np.int16},
            {'groupby': ['channel'], 'select': 'os', 'agg': pd.Series.nunique, 'agg_name': 'nunique', 'new': 'ch_os_n', 'astype': np.int16},
        ]
        for groupby in group_by_list:
            gp = get_gp_from_dict(data_df, groupby)
            data_df = data_df.merge(gp, on=groupby['groupby'], how='left')
            del gp
            gc.collect()
        print(f"~ In add_day_stat_way() df.cols={data_df.columns.values}")
        self.day_stat_bool = True
        return data_df

    def add_time_interval_stat_way(self, data_df, model_name):
        data_df = self.simplest_way(data_df, model_name)
        group_by_list = [
            {'groupby': ['ip', 'click_time'], 'select': 'app', 'agg': pd.Series.nunique, 'agg_name': 'nunique', 'new': 'iptime_app_n'},
            {'groupby': ['ip', 'click_time'], 'select': 'device', 'agg': pd.Series.nunique, 'agg_name': 'nunique', 'new': 'iptime_device_n'},
            {'groupby': ['ip', 'click_time'], 'select': 'os', 'agg': pd.Series.nunique, 'agg_name': 'nunique', 'new': 'iptime_os_n'},
            {'groupby': ['ip', 'click_time'], 'select': 'channel', 'agg': pd.Series.nunique, 'agg_name': 'nunique', 'new': 'iptime_ch_n'},
            {'groupby': ['ip', 'click_time'], 'select': 'channel', 'agg': 'count', 'agg_name': 'count', 'new': 'iptime_click_n'},

            {'groupby': ['ip', 'click_time', 'channel'], 'select': 'app', 'agg': 'count', 'agg_name': 'count', 'new': 'iptimech_click_n'},

            {'groupby': ['ip', 'click_time'], 'select': 'app', 'agg': get_value_counts_entroy, 'agg_name': 'entropy', 'new': 'iptime_app_ent'},
            {'groupby': ['ip', 'click_time'], 'select': 'device', 'agg': get_value_counts_entroy, 'agg_name': 'entropy', 'new': 'iptime_device_ent'},
            {'groupby': ['ip', 'click_time'], 'select': 'os', 'agg': get_value_counts_entroy, 'agg_name': 'entropy', 'new': 'iptime_os_ent'},
            {'groupby': ['ip', 'click_time'], 'select': 'channel', 'agg': get_value_counts_entroy, 'agg_name': 'entropy', 'new': 'iptime_ch_ent'},
        ]
        for groupby in group_by_list:
            gp = get_gp_from_dict(data_df, groupby)
            data_df = data_df.merge(gp, on=groupby['groupby'], how='left')
            del gp
            gc.collect()
        print(f"~ In add_time_interval_stat_way() df.cols={data_df.columns.values}")
        return data_df

    def add_attributed_stat_way(self, data_df, model_name):
        data_df = self.simplest_way(data_df, model_name)
        data_df = self.add_time_interval_stat_way(data_df, model_name)
        group_by_list = [
            {'groupby': ['app'], 'select': 'is_attributed', 'agg': rate_calculation, 'agg_name': 'attributed_rate', 'new': 'appday_attr_rate'},
        ]
        for groupby in group_by_list:
            gp = get_gp_from_dict(data_df, groupby)
            data_df = data_df.merge(gp, on=groupby['groupby'], how='left')
            del gp
            gc.collect()
        print(f"~ In add_attributed_stat_way() df.cols={data_df.columns.values}")
        return data_df

    def make_emb_max(self, train_feat_df):
        if self.simplest_bool:
            self.max_ip = train_feat_df['ip'].max()
            self.max_app = train_feat_df['app'].max()
            self.max_device = train_feat_df['device'].max()
            self.max_os = train_feat_df['os'].max()
            self.max_channel = train_feat_df['channel'].max()
            self.max_click_time = train_feat_df['click_time'].max()
        if self.day_stat_bool:
            self.max_device_os_n = train_feat_df['device_os_n'].max()
            self.max_app_ch_n = train_feat_df['app_ch_n'].max()
            self.max_device_ch_n = train_feat_df['device_ch_n'].max()
            self.max_os_ch_n = train_feat_df['os_ch_n'].max()
            self.max_ch_os_n = train_feat_df['ch_os_n'].max()

    def get_emb_max(self, test_df):
        if self.simplest_bool:
            self.max_ip = update_max(self.max_ip, test_df, 'ip')
            self.max_app = update_max(self.max_app, test_df, 'app')
            self.max_device = update_max(self.max_device, test_df, 'device')
            self.max_os = update_max(self.max_os, test_df, 'os')
            self.max_channel = update_max(self.max_channel, test_df, 'channel')
            self.max_click_time = update_max(self.max_click_time, test_df, 'click_time')
        if self.day_stat_bool:
            self.max_device_os_n = update_max(self.max_device_os_n, test_df, 'device_os_n')
            self.max_app_ch_n = update_max(self.max_app_ch_n, test_df, 'app_ch_n')
            self.max_device_ch_n = update_max(self.max_device_ch_n, test_df, 'device_ch_n')
            self.max_os_ch_n = update_max(self.max_os_ch_n, test_df, 'os_ch_n')
            self.max_ch_os_n = update_max(self.max_ch_os_n, test_df, 'ch_os_n')
            Logger.info(f"各特征取值上限为: "
                        f"\nmax_ip={self.max_ip}"
                        f"\nmax_app={self.max_app}"
                        f"\nmax_device={self.max_device}"
                        f"\nmax_os={self.max_os}"
                        f"\nmax_channel={self.max_channel}"
                        f"\nmax_click_time={self.max_click_time}"
                        f"\nmax_device_os_n={self.max_device_os_n}"
                        f"\nmax_app_ch_n={self.max_app_ch_n}"
                        f"\nmax_device_ch_n={self.max_device_ch_n}"
                        f"\nmax_os_ch_n={self.max_os_ch_n}"
                        f"\nmax_ch_os_n={self.max_ch_os_n}")









"""
DataReader is a class which is used to read dataset and return train/validation/test dataframe.
"""

import pandas as pd
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


class DataReader:

    def __init__(self, file_from:str, feats_construct:str):
        self.file_from = file_from
        self.feats_construct = feats_construct
        # For one train_df will use StratifiedKFold to split train and validation
        # For train_df list each element will be loop as train and validation
        self.train_df_list = []
        self.test_df = None
        self.int_cols = ['ip', 'app', 'device', 'os', 'channel', 'is_attributed']
        self.target = 'is_attributed'
        self.time_cols = ['click_time', 'attributed_time']

    def load_train(self):
        if self.file_from == 'by_day__by_test_time':
            for day in range(7, 10):
                day_path = '../data/day_with_test_hour/201711{:02d}_test_hour.csv'.format(day)
                day_df = pd.read_csv(day_path, dtype=DTYPES)
                self.train_df_list.append(day_df)
        else:
            print(f"!!! Wrong param['file_from'] = '{self.file_from}'")

    def load_test(self):
        self.test_df = pd.read_csv('../data/test.csv', dtype=DTYPES)

    def construct_feats(self, data_df: pd.DataFrame):
        if self.feats_construct == 'simplest':
            for col in self.int_cols:
                if col in data_df.columns:
                    data_df[col] = data_df[col].astype('category')
            data_df['click_time'] = pd.to_datetime(data_df['click_time'])
            data_df['click_time'] = data_df['click_time'].map(lambda t: datetime2cate('test_30mins', t))
            data_df['click_time'] = data_df['click_time'].astype('category')
            cols = self.int_cols[:5] + ['click_time', 'is_attributed'] if self.target in data_df.columns \
                else self.int_cols[:5] + ['click_time']
            return data_df[cols]
        else:
            print(f"!!! Wrong param['feats_construct'] = '{self.feats_construct}'")

    def get_train_feats_df(self):
        with timer("Loading train csv files"):
            self.load_train()
        train_feat_df = pd.DataFrame()
        each_len = []
        for train_df in self.train_df_list:
            with timer(f"Construct train feats df<'{self.feats_construct}'>"):
                each_len.append(train_df.shape[0])
                train_feat_df = pd.concat([train_feat_df, self.construct_feats(train_df)], axis=0, ignore_index=True)
                del train_df
                gc.collect()
        self.train_df_list = []
        cv_index_list = []  # [(train_idx, test_idx), (train_idx, test_idx), ...]
        index_start = 0
        for len_ in each_len:
            train_idx = train_feat_df.index[0: index_start].tolist() + train_feat_df.index[index_start+len_:].tolist()
            test_idx = train_feat_df.index[index_start: index_start + len_].tolist()
            cv_index_list.append((train_idx, test_idx))
            index_start += len_
        return train_feat_df, cv_index_list, self.target

    def get_test_feats_df(self):
        with timer("Loading test csv file"):
            self.load_test()
        with timer(f"Construct test feats df<'{self.feats_construct}'>"):
            test_df = self.construct_feats(self.test_df)
            del self.test_df
            gc.collect()
        return test_df







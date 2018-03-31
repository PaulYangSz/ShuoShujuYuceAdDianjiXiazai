"""
Use click_time to split the train dataset.
"""

import pandas as pd
from datetime import datetime, timedelta
import gc


import logging
import logging.config


def start_logging():
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

DTYPES = {
    'ip': 'uint32',
    'app': 'uint16',
    'device': 'uint16',
    'os': 'uint16',
    'channel': 'uint16',
    'is_attributed': 'uint8',
    'click_id': 'uint32',
}


def analysis_test_hour_distrib(test_path):
    test_df = pd.read_csv(test_path, dtype=DTYPES)
    Logger.info(f"test_df[{test_path}].shape={test_df.shape}")
    Logger.info(f"Int类型test_df[{test_path}].describe:\n{test_df.describe()}")
    variables = ['ip', 'app', 'device', 'os', 'channel']
    for v in variables:
        test_df[v] = test_df[v].astype('category')
    test_df['click_time'] = pd.to_datetime(test_df['click_time'])
    Logger.info(f"Cate&Date类型test_df[{test_path}].describe:\n{test_df[variables + ['click_time']].describe()}")
    test_df['click_rnd_H'] = test_df.click_time.dt.round('H')
    test_df['click_rnd_H'] = test_df['click_rnd_H'].map(lambda x: x.hour)
    Logger.info(f"Round Hour Distribution:\n{test_df.click_rnd_H.value_counts().sort_index()}")
    test_df['click_org_H'] = test_df.click_time.map(lambda x: x.hour)
    Logger.info(f"Original Hour Distribution:\n{test_df['click_org_H'].value_counts().sort_index()}")
    Logger.info("继续查看时间段的截取依据")
    group_H_ = test_df[['click_rnd_H', 'click_time']].groupby(['click_rnd_H'], as_index=True)
    rnd_H_ser_min = group_H_.min()
    rnd_H_ser_min.rename(columns={'click_time': 'min'}, inplace=True)
    rnd_H_ser_max = group_H_.max()
    rnd_H_ser_max.rename(columns={'click_time': 'max'}, inplace=True)
    Logger.info(f"查看下三个时间段的取值范围:\n{pd.concat([rnd_H_ser_min, rnd_H_ser_max], axis=1)}")
    Logger.info("可以看出test数据是取自04:00——06:00,09:00——11:00,13:00——15:00")
    Logger.info("那么是否应该也在train数据集中截取相同时间段的数据来训练？")
    del test_df
    gc.collect()


def glance_train_data():
    train_path = '../data/train.csv'
    train_df = pd.read_csv(train_path)
    Logger.info(f"不使用DTYPES读取, info():\n{train_df.info()}")
    del train_df
    gc.collect()
    train_df = pd.read_csv(train_path, dtype=DTYPES)
    Logger.info(f"使用DTYPES读取, info():\n{train_df.info()}")
    Logger.info(f"train_df.shape={train_df.shape}")
    Logger.info(f"查看Int类型的describe():\n{train_df.describe()}")
    variables = ['ip', 'app', 'device', 'os', 'channel', 'is_attributed']
    for v in variables:
        train_df[v] = train_df[v].astype('category')
    train_df['click_time'] = pd.to_datetime(train_df['click_time'])
    train_df['attributed_time'] = pd.to_datetime(train_df['attributed_time'])
    Logger.info(f"查看Cate类型&Date类型的describe():\n{train_df.describe()}")
    Logger.info(f"Click的最早时间{train_df['click_time'].min()}")
    Logger.info(f"Click的最晚时间{train_df['click_time'].max()}")
    Logger.info(f"Attrib的最早时间{train_df['attributed_time'].min()}")
    Logger.info(f"Attrib的最晚时间{train_df['attributed_time'].max()}")
    del train_df
    gc.collect()


def split_train_by_day():
    train_path = '../data/train.csv'
    train_df = pd.read_csv(train_path, dtype=DTYPES)
    for day in range(6, 10):
        day_str = f'2017-11-0{day}'
        in_this_day = train_df['click_time'].map(lambda x: x.startswith(day_str)).values
        day_df = train_df[in_this_day]
        train_df = train_df[~in_this_day]
        day_path = '../data/days/201711{:02d}.csv'.format(day)
        day_df.to_csv(day_path, index=False)
        del day_df
        gc.collect()


def split_train_by_dayandtesttime():
    def check_in_test_time(item_time: datetime, _test_times):
        in_flag = False
        for i in range(0, 5, 2):
            if _test_times[i] <= item_time <= _test_times[i + 1]:
                in_flag = True
                break
        return in_flag

    for day in range(6, 10):
        test_times = [datetime(2017, 11, day, 4), datetime(2017, 11, day, 6),
                      datetime(2017, 11, day, 9), datetime(2017, 11, day, 11),
                      datetime(2017, 11, day, 13), datetime(2017, 11, day, 15)]
        day_path = '../data/days/201711{:02d}.csv'.format(day)
        train_day_df = pd.read_csv(day_path, dtype=DTYPES)
        Logger.info(f"train_day_df[{day_path}].shape={train_day_df.shape}")
        Logger.info(f"Int类型train_day_df[{day_path}].describe:\n{train_day_df.describe()}")
        variables = ['ip', 'app', 'device', 'os', 'channel', 'is_attributed']
        for v in variables:
            train_day_df[v] = train_day_df[v].astype('category')
        train_day_df['click_time'] = pd.to_datetime(train_day_df['click_time'])
        train_day_df['attributed_time'] = pd.to_datetime(train_day_df['attributed_time'])
        Logger.info(f"Cate&Date类型test_df[{day_path}].describe:\n{train_day_df[variables + ['click_time']].describe()}")
        train_day_df['click_org_H'] = train_day_df.click_time.map(lambda x: x.hour)
        Logger.info(f"Original Hour Distribution:\n{train_day_df['click_org_H'].value_counts().sort_index()}")
        train_day_df = train_day_df[train_day_df['click_time'].map(lambda x: check_in_test_time(x, test_times))]
        day_path = '../data/day_with_test_hour/201711{:02d}_test_hour.csv'.format(day)
        train_day_df['click_time'] = train_day_df['click_time'].map(str)
        train_day_df['attributed_time'] = train_day_df['attributed_time'].map(str)
        train_day_df.to_csv(day_path, index=False)
        del train_day_df
        gc.collect()


if __name__ == '__main__':
    pd.set_option('display.max_rows', 100)
    pd.set_option('display.max_columns', 100)
    pd.set_option('display.width', 10000)
    analysis_test_hour_distrib('../data/test.csv')
    analysis_test_hour_distrib('../data/test_supplement.csv')
    glance_train_data()

    split_method = 'by_day__by_test_time'
    if split_method == "by_day":
        split_train_by_day()
    elif split_method == 'by_day__by_test_time':
        split_train_by_dayandtesttime()
    else:
        pass




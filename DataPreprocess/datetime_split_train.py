"""
Use click_time to split the train dataset.
"""

import pandas as pd


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


def analysis_test_hour_distrib():
    test_df = pd.read_csv('../data/test.csv', dtype=DTYPES)
    test_df['click_time'] = pd.to_datetime(test_df['click_time'])
    test_df['click_rnd_H'] = test_df.click_time.dt.round('H')
    test_df['click_rnd_H'] = test_df['click_rnd_H'].map(lambda x: x.hour)
    Logger.info(f"Round Hour Distribution:\n{test_df.click_rnd_H.value_counts().sort_index()}")
    test_df['click_org_H'] = test_df.click_time.map(lambda x: x.strftime('%H'))
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


def split_train_by_dayandtesttime():
    pass


if __name__ == '__main__':
    analysis_test_hour_distrib()

    split_method = 'by_day__by_test_time'
    if split_method == 'by_day__by_test_time':
        split_train_by_dayandtesttime()


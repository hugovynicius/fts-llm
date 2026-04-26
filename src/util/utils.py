from typing import List
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset
import numpy as np
import warnings

from torch.utils.data import DataLoader

from src.util.Dataset_Custom import Dataset_Custom
from src.util.Dataset_ETT_hour import Dataset_ETT_hour
from src.util.Dataset_ETT_minute import Dataset_ETT_minute
from src.util.Dataset_sin import Dataset_sin
from src.util.DayOfMonth import DayOfMonth
from src.util.DayOfWeek import DayOfWeek
from src.util.DayOfYear import DayOfYear
from src.util.HourOfDay import HourOfDay
from src.util.MinuteOfHour import MinuteOfHour
from src.util.MonthOfYear import MonthOfYear
from src.util.SecondOfMinute import SecondOfMinute
from src.util.TimeFeature import TimeFeature
from src.util.WeekOfYear import WeekOfYear

warnings.filterwarnings('ignore')

def time_features_from_frequency_str(freq_str: str) -> List[TimeFeature]:
    """
    Returns a list of time features that will be appropriate for the given frequency string.
    Parameters
    ----------
    freq_str
        Frequency string of the form [multiple][granularity] such as "12H", "5min", "1D" etc.
    """

    features_by_offsets = {
        offsets.YearEnd: [],
        offsets.QuarterEnd: [MonthOfYear],
        offsets.MonthEnd: [MonthOfYear],
        offsets.Week: [DayOfMonth, WeekOfYear],
        offsets.Day: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.BusinessDay: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Hour: [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Minute: [
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
        offsets.Second: [
            SecondOfMinute,
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
    }

    offset = to_offset(freq_str)

    for offset_type, feature_classes in features_by_offsets.items():
        if isinstance(offset, offset_type):
            return [cls() for cls in feature_classes]

    supported_freq_msg = f"""
    Unsupported frequency {freq_str}
    The following frequencies are supported:
        Y   - yearly
            alias: A
        M   - monthly
        W   - weekly
        D   - daily
        B   - business days
        H   - hourly
        T   - minutely
            alias: min
        S   - secondly
    """
    raise RuntimeError(supported_freq_msg)


def time_features(dates, freq='h'):
    return np.vstack([feat(dates) for feat in time_features_from_frequency_str(freq)])

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'sin':Dataset_sin,
}


def data_provider(args, flag):
    Data = data_dict[args['name_dataset']]
    timeenc = 0 if 'timeF' != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = 128
        freq = 'h'
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = 'h'
        Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = 128
        freq = 'h'

    data_set = Data(
        root_path=args['root_path'],
        data_path=args['path'],
        flag=flag,
        size=[96, 48, 96],
        features='M',
        target='OT',
        timeenc=timeenc,
        freq=freq
    )

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=10,
        drop_last=drop_last)
    return data_set, data_loader

def get_data(name_dataset, root_path, path, flag):
    args = {
        'name_dataset': name_dataset,
        'root_path': root_path,
        'path': path
    }
    data_set, data_loader = data_provider(args, flag)
    return data_set, data_loader



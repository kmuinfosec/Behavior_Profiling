import os

import numpy as np
from datetime import datetime

datetime_format = '%Y-%m-%d %H:%M:%S.%f'


def get_int_time(start_time):
    ts_timestamp = datetime.strptime(start_time, datetime_format).timestamp()
    return ts_timestamp


def get_str_time(int_time):
    return datetime.fromtimestamp(int_time).strftime(datetime_format)


def get_int(st_list, end_list):
    if len(st_list) == 1:
        return 0
    else:
        int_list = []
        for i in range(len(end_list)-1):
            int_list.append(get_int_time(end_list[i]) - get_int_time(st_list[i+1]))
        return int_list


def make_col_index(data_path):
    data = os.listdir(data_path)[0]
    with open(rf"{data_path}\{data}", 'r', encoding='utf-8') as f:
        tmp = f.readline().strip().split(',')
        column_index = {i : idx for idx, i in enumerate(tmp)}
        return column_index


def gini(value_list):
    uniq_value, count = np.unique(value_list, return_counts=True)
    gini_value = 1

    for c in count:
        gini_value -= (c/sum(count)) ** 2

    return gini_value


def format_date(precision):
    format_order = ["%Y", "%m", "%d", "%H", "%M", "%S"]
    format_str = ""
    for i, fmt in enumerate(format_order):
        if i == precision + 1:
            break
        else:
            format_str += fmt
            if i == 2:
                format_str += '_'

    return datetime.today().strftime(format_str)


def get_time_window(start_time, time_window):
    datetime_format = '%Y-%m-%d %H:%M:%S'
    ts_timestamp = datetime.strptime(start_time, datetime_format).timestamp()
    window_start = ts_timestamp - (ts_timestamp % time_window)
    window_end = window_start + time_window
    window_start_str = datetime.fromtimestamp(window_start).strftime(datetime_format)
    window_end_str = datetime.fromtimestamp(window_end).strftime(datetime_format)

    return window_start_str, window_end_str


def init_config(options):
    config = {}
    config['mode'] = options.mode if options.mode else 'count'
    config['method'] = options.mode if options.method else 'hybrid'
    config['min_sample'] = options.min_sample if options.min_sample else 5
    config['timeout'] = options.timeout if options.timeout else 3600
    config['label_set'] = options.label_set if options.label_set else 'Abused'
    config['hybrid_count'] = options.hybrid_count if options.hybrid_count else 3
    config['abused_score'] = options.abused if options.abused else 90
    config['k'] = options.k if options.k else 100
    config['save_rce'] = bool(options.save_rce)
    config['output_method'] = options.output_method if options.output_method else 'min'
    if options.preprocessing_path:
        config['preprocessing_path'] = os.path.abspath(options.preprocessing_path)
    else:
        config['preprocessing_path'] = False
    return config

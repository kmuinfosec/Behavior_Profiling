import os
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
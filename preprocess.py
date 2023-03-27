import os
import pickle
from ipaddress import IPv4Address
from feature.feature_extracting import run_profiling

import numpy as np

from sklearn.preprocessing import MinMaxScaler


def parse_config_file(file_path, target='kookmin'):
    with open(file_path, 'r') as f:
        lines = list(map(str.strip, f.readlines()))

    content_list = []

    is_find = False
    for line in lines:
        if not is_find:
            if line == '[' + target + ']':
                is_find = True
        else:
            if line.startswith('[') and line.endswith(']'):
                break
            if line:
                content_list.append(line)
    return content_list


def inflate_subnet_ip(inside_ip_list):
    total_ip_set = set()
    for inside_ip in inside_ip_list:
        if '/' in inside_ip:
            inside_ip, cidr = inside_ip.split('/')
            start_address = int(IPv4Address(inside_ip))
            end_address = int(IPv4Address(inside_ip)) + 2 ** (32 - int(cidr)) - 1
            for subnet_address in range(start_address, end_address + 1):
                total_ip_set.add(str(IPv4Address(subnet_address)))
        else:
            total_ip_set.add(inside_ip)
    return total_ip_set


def load_data(min_sample, timeout, data_path, phase, score_dict):
    if os.path.exists(rf"{data_path}\min_{min_sample}_to_{timeout}_feature_{phase}.pkl"):
        with open(rf"{data_path}\min_{min_sample}_to_{timeout}_feature_{phase}.pkl", 'rb') as f:
            tmp = pickle.load(f)
        with open(rf"{data_path}\min_{min_sample}_to_{timeout}_key_{phase}.pkl", 'rb') as f:
            key = pickle.load(f)
    else:
        tmp = np.load(rf"{data_path}\over_min_{min_sample}_to_{timeout}_feature_{phase}.npy")
        key = np.load(rf"{data_path}\over_min_{min_sample}_to_{timeout}_key_{phase}.npy")

    data = np.array(tmp).T.tolist()
    score = [score_dict[ip][0] if ip in score_dict else 0 for ip in tmp[0]]
    label = [1 if score >= 90 else 0 for score in score]
    return data, label, key


def load_raw(label_set):
    if label_set == 'Abused':
        with open(r"C:\jupyter_project\NCSC2023\Dataset\ip_score_date.pkl", 'rb') as f:
            score_dict = pickle.load(f)
    elif label_set == 'GN':
        with open(r"C:\jupyter_project\NCSC2023\Dataset\GN_Label_dict.pickle", 'rb') as f:
            score_dict = pickle.load(f)

    train_path_list = parse_config_file(r'./config/train_dir.txt')
    test_path_list = parse_config_file(r'./config/test_dir.txt')
    inside_ip_list = parse_config_file(r'./config/inside_list.txt')
    inside_ip_set = inflate_subnet_ip(inside_ip_list)
    return train_path_list, test_path_list, inside_ip_set, score_dict


def preprocessing(min_sample, timeout, train_path, test_path, inside_ip_set, score_dict):
    print("Train Profiling")
    train_data, train_label, train_key, train_stat = run_profiling(train_path, inside_ip_set, min_sample, timeout,
                                                                   score_dict, 90, 'count')
    print("Test Profiling")
    test_data, test_label, test_key, test_stat = run_profiling(test_path, inside_ip_set, min_sample, timeout,
                                                               score_dict, 90, 'count')

    print("Min Sample", min_sample, "Timeout :", timeout)
    print("Train Data Size :", len(train_data), "[0]", train_label.count(0), "[1]", train_label.count(1))
    print("Test Data Size :", len(test_data), "[0]", test_label.count(0), "[1]", test_label.count(1))

    print("Data Scaling...")
    scaler = MinMaxScaler()
    train_idx = np.where(np.array(train_label) == 1)
    train_x = scaler.fit_transform(np.array(train_data)[train_idx[0], 1:-1])
    test_x = scaler.transform(np.array(test_data)[:, 1:-1])
    test_ip_list = np.array(test_data)[:, 0]
    return train_x, test_x, test_label, test_ip_list

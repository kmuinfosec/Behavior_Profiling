import pickle
import sys

from feature.profiling import CountBasedProfile, TimeBasedProfile


def cal_stat(stat_dict, label_list, min_data):
    ip_list = stat_dict.keys()
    attack_size_list, attack_max_list, benign_size_list, benign_max_list = [], [], [], []
    count_ip_dict = {"BEN": {"pos": 0, "neg": 0}, "MAL": {"pos": 0, "neg": 0}}
    count_data_dict = {"BEN": {"pos": 0, "neg": 0}, "MAL": {"pos": 0, "neg": 0}}

    for idx, ip in enumerate(ip_list):
        if label_list[idx] == 1:
            attack_size_list += stat_dict[ip]
            attack_max_list.append(max(stat_dict[ip]))
            if max(stat_dict[ip]) == min_data:
                count_ip_dict['MAL']['pos'] += 1
            else:
                count_ip_dict['MAL']['neg'] += 1

            for c in stat_dict[ip]:
                if c == min_data:
                    count_data_dict['MAL']['pos'] += 1
                else:
                    count_data_dict['MAL']['neg'] += 1
        else:
            benign_size_list += stat_dict[ip]
            benign_max_list.append(max(stat_dict[ip]))
            if max(stat_dict[ip]) == min_data:
                count_ip_dict['BEN']['pos'] += 1
            else:
                count_ip_dict['BEN']['neg'] += 1

            for c in stat_dict[ip]:
                if c == min_data:
                    count_data_dict['BEN']['pos'] += 1
                else:
                    count_data_dict['BEN']['neg'] += 1

    print("Total_Benign_IP :", label_list.count(0), "Pos_Benign_IP :", count_ip_dict['BEN']['pos'])
    print("Total_Mal_IP :", label_list.count(1), "Pos_Mal_IP :", count_ip_dict['MAL']['pos'])
    print("Pos_benign_data :", count_data_dict['BEN']['pos'], "Pos_mal_data :", count_data_dict['MAL']['pos'])
    print("Neg_benign_data :", count_data_dict['BEN']['neg'], "Neg_mal_data :", count_data_dict['MAL']['neg'])


def run_profiling(data_path, inside_ip_set, min_sample, timeout, score_dict, label_score=90, method='count'):
    if method == 'time':
        profiler = TimeBasedProfile(data_path, inside_ip_set, min_sample, timeout, 'discard')
    else:
        profiler = CountBasedProfile(data_path, inside_ip_set, min_sample, timeout, 'discard')

    profiler.profiling()
    data_list = profiler.get_matrix()
    key_list = profiler.get_keys()
    label = []
    for key in key_list:
        ip, st, et = key.split('_')
        if ip not in score_dict:
            label.append(0)
        else:
            label.append(0 if score_dict[ip][0] < label_score else 1)
    stat = profiler.get_stat()
    cal_stat(stat, label, min_sample)

    return data_list, label, key_list, stat



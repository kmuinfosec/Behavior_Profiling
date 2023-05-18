import pickle
import sys

from feature.profiling import CountBasedProfile, TimeBasedProfile


def cal_stat(stat_dict, score_dict, config):
    ip_list = stat_dict.keys()
    count_ip_dict = {"BEN": {"pos": 0, "neg": 0}, "MAL": {"pos": 0, "neg": 0}}
    count_data_dict = {"BEN": {"pos": 0, "neg": 0}, "MAL": {"pos": 0, "neg": 0}}
    count_total_ip_dict = {"BEN" : 0, "MAL" : 0}
    for idx, ip in enumerate(ip_list):
        if ip in score_dict and score_dict[ip][0][0] >= config['abused_score']:
            count_total_ip_dict['MAL'] += 1
            if len(stat_dict[ip]) > 0 and max(stat_dict[ip]) >= config['hybrid_count']:
                count_ip_dict['MAL']['pos'] += 1
            else:
                count_ip_dict['MAL']['neg'] += 1

            for c in stat_dict[ip]:
                if c >= config['hybrid_count']:
                    count_data_dict['MAL']['pos'] += 1
                else:
                    count_data_dict['MAL']['neg'] += 1
        else:
            count_total_ip_dict['BEN'] += 1
            if len(stat_dict[ip]) > 0 and max(stat_dict[ip]) >= config['hybrid_count']:
                count_ip_dict['BEN']['pos'] += 1
            else:
                count_ip_dict['BEN']['neg'] += 1

            for c in stat_dict[ip]:
                if c >= config['hybrid_count']:
                    count_data_dict['BEN']['pos'] += 1
                else:
                    count_data_dict['BEN']['neg'] += 1

    print("Total_Benign_IP :", count_total_ip_dict['BEN'], "Pos_Benign_IP :", count_ip_dict['BEN']['pos'])
    print("Total_Mal_IP :", count_total_ip_dict['MAL'], "Pos_Mal_IP :", count_ip_dict['MAL']['pos'])
    print("Pos_benign_data :", count_data_dict['BEN']['pos'], "Pos_mal_data :", count_data_dict['MAL']['pos'])
    print("Neg_benign_data :", count_data_dict['BEN']['neg'], "Neg_mal_data :", count_data_dict['MAL']['neg'])


def run_profiling(phase, data_path, inside_ip_set, score_dict, config):
    if config['preprocessing_path']:
        with open(rf"{config['preprocessing_path']}\{phase}_feature.pkl", 'rb') as f:
            data_list = pickle.load(f)
        with open(rf"{config['preprocessing_path']}\{phase}_key.pkl", 'rb') as f:
            key_list = pickle.load(f)
        with open(rf"{config['preprocessing_path']}\{phase}_stat.pkl", 'rb') as f:
            stat = pickle.load(f)
    else:
        if config['mode'] == 'time':
            profiler = TimeBasedProfile(data_path, inside_ip_set, config)
        else:
            profiler = CountBasedProfile(data_path, inside_ip_set, config)
        profiler.profiling()
        data_list = profiler.get_matrix()
        key_list = profiler.get_keys()
        stat = profiler.get_stat()
        with open(rf"{config['save_path']}\{phase}_feature.pkl", 'wb') as f:
            pickle.dump(data_list, f)
        with open(rf"{config['save_path']}\{phase}_key.pkl", 'wb') as f:
            pickle.dump(key_list, f)
        with open(rf"{config['save_path']}\{phase}_stat.pkl", 'wb') as f:
            pickle.dump(stat, f)
    label = []
    for key in key_list:
        tmp = key.split('_')
        if len(tmp) == 3:
            ip, st, et = tmp
        else:
            ip, _, st, et = tmp
        if ip in score_dict and score_dict[ip][0][0] >= config['abused_score']:
            label.append(1)
        else:
            label.append(0)
    cal_stat(stat, score_dict, config)

    return data_list, label, key_list, stat



import os

import numpy as np

from tqdm import tqdm
from utils import get_int_time, get_str_time, make_col_index, get_time_window

from feature.profile import Profile
from feature.abstract_profile import CommonProfile


class CountBasedProfile(CommonProfile):
    def __init__(self, data_path, inside_ip_set, min_sample, timeout, method='hybrid', hybrid=3):
        super(CountBasedProfile, self).__init__()
        self.hybrid = hybrid
        self.method = method
        self.data_path = data_path
        self.min_sample = min_sample
        self.timeout = timeout
        self.inside_ip_set = inside_ip_set
        self.feature_matrix = [[] for _ in range(len(self.feature_list))]
        self.profile_key_list = []
        self.profile_list = []
        self.column_index = make_col_index(self.data_path[0])
        self.flow_stack = {}
        self.stat_dict = {}

    def profiling(self):
        for folder in self.data_path:
            for file in os.listdir(folder):
                print(f"{file} extracting...")
                with open(rf"{folder}\{file}", "r", encoding='utf-8') as f:
                    col = f.readline().strip().split(',')
                    column_index = {i: idx for idx, i in enumerate(col)}
                    for tmp_flow in tqdm(f.readlines()):
                        flow = tmp_flow.strip().split(',')
                        target_ip = self.find_target_ip(flow)
                        if target_ip is None:
                            continue
                        now_time = get_int_time(flow[column_index['first']])

                        if target_ip not in self.flow_stack:
                            self.process_new_ip(target_ip, now_time)
                        while self.flow_stack[target_ip]['st_time'] < now_time - self.timeout:
                            self.process_timeout(target_ip, now_time)
                            self.process_new_ip(target_ip, now_time)
                        self.grouping_flow(flow, now_time, target_ip)
        print(f"Profiling Finish...")
        self.process_rest_dict()
        self.make_feature()

    def find_target_ip(self, flow):
        sip, dip = flow[self.column_index['source']], flow[self.column_index['destination']]
        if sip[-1] != '_' and dip[-1] != '_':
            return None
        elif sip[-1] == '_' and dip[-1] != '_':
            target_ip = dip
        else:
            target_ip =  sip
        # if sip not in self.inside_ip_set and dip not in self.inside_ip_set:
        #     return None
        # elif sip in self.inside_ip_set and dip not in self.inside_ip_set:
        #     target_ip = dip
        # else:
        #     target_ip = sip
        return target_ip

    def process_rest_dict(self):
        for key in list(self.flow_stack.keys()):
            self.process_timeout(key, 1e+10)

    def make_feature(self):
        for profile in tqdm(self.profile_list):
            for i, feature in enumerate(self.feature_list):
                self.feature_matrix[i].append(self.feature_func_map[feature](profile))
        self.feature_matrix = np.array(self.feature_matrix).T.tolist()

    def grouping_flow(self, flow, now_time, target_ip):
        self.flow_stack[target_ip]['flow'].append(flow)
        self.flow_stack[target_ip]['end_time'] = now_time
        if len(self.flow_stack[target_ip]['flow']) == self.min_sample:
            self.update_flow(target_ip)
            self.flow_stack[target_ip]['st_time'] = get_int_time(
                self.flow_stack[target_ip]['flow'][0][self.column_index['first']])

    def update_flow(self, target_ip):
        self.stat_dict[target_ip].append(self.min_sample)
        profile, profile_key = self.make_profile(self.flow_stack[target_ip]['flow'], target_ip,
                                                 self.flow_stack[target_ip]['st_time'],
                                                 self.flow_stack[target_ip]['end_time'])
        self.profile_list.append(profile)
        self.profile_key_list.append(profile_key)
        self.flow_stack[target_ip]['flow'].pop(0)

    def process_timeout(self, target_ip, now_time):
        if self.method == 'rest':
            self.update_flow(target_ip)
        elif self.method == 'one_rest':
            self.update_flow(target_ip)
            del self.flow_stack[target_ip]
        elif self.method == 'discard':
            self.stat_dict[target_ip].append(len(self.flow_stack[target_ip]['flow']))
            del self.flow_stack[target_ip]
        elif self.method == 'hybrid':
            self.stat_dict[target_ip].append(len(self.flow_stack[target_ip]['flow']))
            if len(self.flow_stack[target_ip]['flow']) >= self.hybrid:
                self.update_flow(target_ip)
            else:
                self.flow_stack[target_ip]['flow'].pop(0)

    def process_new_ip(self, target_ip, now_time):
        if target_ip not in self.flow_stack or len(self.flow_stack[target_ip]['flow']) == 0:
            self.flow_stack[target_ip] = {'flow': []}
            self.flow_stack[target_ip]['st_time'] = now_time
            self.stat_dict[target_ip] = []
        else:
            self.flow_stack[target_ip]['st_time'] = get_int_time(
                self.flow_stack[target_ip]['flow'][0][self.column_index['first']])

    def make_profile(self, flow_list, target_ip, st_time, end_time):
        profile_key = '{}_{}_{}'.format(target_ip, get_str_time(st_time), get_str_time(end_time))
        new_pf = Profile(profile_key)
        for flow in flow_list:
            new_pf.add(self.add_flow(flow))
        return new_pf, profile_key

    def add_flow(self, flow):
        sip, dip = flow[self.column_index['source']], flow[self.column_index['destination']]
        attr_map = {}
        if self.profiling_target == 'destination':
            # if sip in self.inside_ip_set:
            if dip[-1] != '_':
                attr_map = self.attribute_map
            else:
                attr_map = self.attribute_map_inv
        elif self.profiling_target == 'source':
            # if dip in self.inside_ip_set:
            if sip[-1] != '_':
                attr_map = self.attribute_map_inv
            else:
                attr_map = self.attribute_map

        attr_dict = {}
        for attr, column in attr_map.items():
            attr_dict[attr] = flow[self.column_index[column]]
        return attr_dict

    def get_matrix(self):
        return self.feature_matrix

    def get_keys(self):
        return self.profile_key_list

    def get_stat(self):
        return self.stat_dict


class TimeBasedProfile(CommonProfile):
    def __init__(self, data_path, inside_ip_set, min_sample, time_window, method='discard'):
        super(TimeBasedProfile, self).__init__()
        self.min_sample = min_sample
        self.time_window = time_window
        self.method = method
        self.data_path = data_path
        self.inside_ip_set = inside_ip_set
        self.feature_matrix = [[] for _ in range(len(self.feature_list))]
        self.profile_key_list = []
        self.profile_list = []
        self.column_index = make_col_index(self.data_path[0])
        self.profile_index = {}
        self.profile_index_inv = {}
        self.profile_cnt = 0

    def profiling(self):
        for folder in self.data_path:
            for file in os.listdir(folder):
                print(f"{file} extracting...")
                with open(rf"{folder}\{file}", 'r', encoding='utf-8') as f:
                    f.readline()  # pass column row
                    flow_list = f.readlines()

                for flow in flow_list:
                    flow = list(map(str.strip, flow.strip().split(",")))
                    self.add_flow(flow)
        self.make_feature()

    def add_flow(self, flow: list):
        sip, dip = flow[self.column_index['sip']], flow[self.column_index['dip']]
        start_time = flow[self.column_index['time_start']]
        window_start, window_end = get_time_window(start_time, self.time_window)

        attr_map = {}
        profile_key = '{}_{}_{}'
        if self.profiling_target == 'dip':
            if dip not in self.inside_ip_set:
                attr_map = self.attribute_map
                profile_key = profile_key.format(dip, window_start, window_end)
            elif sip not in self.inside_ip_set:
                attr_map = self.attribute_map_inv
                profile_key = profile_key.format(sip, window_start, window_end)
            elif dip in self.inside_ip_set and sip in self.inside_ip_set:
                attr_map = self.attribute_map_inv
                profile_key = profile_key.format(sip, window_start, window_end)
            else:
                return
        elif self.profiling_target == 'sip':
            if sip not in self.inside_ip_set:
                attr_map = self.attribute_map_inv
                profile_key = profile_key.format(sip, window_start, window_end)
            elif dip not in self.inside_ip_set:
                attr_map = self.attribute_map
                profile_key = profile_key.format(dip, window_start, window_end)
            elif dip in self.inside_ip_set and sip in self.inside_ip_set:
                attr_map = self.attribute_map_inv
                profile_key = profile_key.format(sip, window_start, window_end)
            else:
                return

        attr_dict = {}
        for attr, column in attr_map.items():
            attr_dict[attr] = flow[self.column_index[column]]
        self.add_profile(profile_key)
        self.profile_list[self.profile_index[profile_key]].add(attr_dict)

    def make_feature(self):
        for profile in tqdm(self.profile_list):
            if self.method == 'discard' & len(profile) < self.min_sample:
                continue
            for i, feature in enumerate(self.feature_list):
                self.feature_matrix[i].append(self.feature_func_map[feature](profile))
        self.feature_matrix = np.array(self.feature_matrix).T.tolist()

    def add_profile(self, profile_key):
        if profile_key not in self:
            new_pf = Profile(profile_key)
            self.profile_list.append(new_pf)
            self.profile_index[profile_key] = self.profile_cnt
            self.profile_index_inv[self.profile_cnt] = profile_key
            self.profile_cnt += 1

    def get_matrix(self):
        return self.feature_matrix

    def get_keys(self):
        return self.profile_key_list

    # def get_stat(self):
    #     return self.stat_dict

import os

import numpy as np

from tqdm import tqdm
from utils import get_int_time, get_str_time, make_col_index

from feature.profile import Profile
from feature.abstract_profile import CommonProfile


class CountBasedProfile(CommonProfile):
    def __init__(self, data_path, inside_ip_set, min_sample, timeout, method='discard'):
        super(CountBasedProfile, self).__init__()
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
                        if self.flow_stack[target_ip]['st_time'] < now_time - self.timeout:
                            self.process_timeout(target_ip, now_time)
                            self.process_new_ip(target_ip, now_time)
                        self.grouping_flow(flow, now_time, target_ip)
        print(f"Profiling Finish...")
        self.process_rest_dict()
        self.make_feature()

    def find_target_ip(self, flow):
        sip, dip = flow[self.column_index['source']], flow[self.column_index['destination']]
        if sip not in self.inside_ip_set and dip not in self.inside_ip_set:
            return None
        elif sip in self.inside_ip_set and dip not in self.inside_ip_set:
            target_ip = dip
        else:
            target_ip = sip
        return target_ip

    def process_rest_dict(self):
        for key in list(self.flow_stack.keys()):
            self.process_timeout(key, 999999999999)

    def make_feature(self):
        for profile in tqdm(self.profile_list):
            for i, feature in enumerate(self.feature_list):
                self.feature_matrix[i].append(self.feature_func_map[feature](profile))
            self.profile_key_list.append(profile.profile_key)
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
            while self.flow_stack[target_ip]['st_time'] < now_time - self.timeout:
                self.update_flow(target_ip)
                if len(self.flow_stack[target_ip]['flow']) == 0:
                    self.flow_stack[target_ip]['st_time'] = now_time
                    break
                else:
                    self.flow_stack[target_ip]['st_time'] = get_int_time(
                        self.flow_stack[target_ip]['flow'][0][self.column_index['first']])
        elif self.method == 'one_rest':
            self.update_flow(target_ip)
            del self.flow_stack[target_ip]
        elif self.method == 'discard':
            self.stat_dict[target_ip].append(len(self.flow_stack[target_ip]['flow']))
            del self.flow_stack[target_ip]
        elif self.method == 'hybrid':
            pass

    def process_new_ip(self, target_ip, now_time):
        self.flow_stack[target_ip] = {'flow': []}
        self.flow_stack[target_ip]['st_time'] = now_time

        self.stat_dict[target_ip] = []

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
            if sip in self.inside_ip_set:
                attr_map = self.attribute_map
            elif dip in self.inside_ip_set:
                attr_map = self.attribute_map_inv
            else:
                return
        elif self.profiling_target == 'source':
            if dip in self.inside_ip_set:
                attr_map = self.attribute_map_inv
            elif sip in self.inside_ip_set:
                attr_map = self.attribute_map
            else:
                return

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
        self.column_index = make_col_index(self.data_path[0])

    def profiling(self):
        profile_list = []
        profile_key_list = []
        flow_stack = {}
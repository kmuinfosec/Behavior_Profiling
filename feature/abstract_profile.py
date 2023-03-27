import numpy as np

from ipaddress import IPv4Address
from utils import get_int
from scipy.stats import entropy


class CommonProfile:
    def __init__(self):
        self.profiling_target = 'destination'

        self.attribute_map = {'target_ip': 'destination', 'target_port': 'dst_port', 'opposite_ip': 'source',
                              'opposite_port': 'src_port', 'duration': 'duration', 'target_pkts': 'out_pkts',
                              'opposite_pkts': 'in_pkts', 'target_bytes': 'out_bytes', 'opposite_bytes': 'in_bytes',
                              'start_time': 'first', 'end_time': 'last'}

        self.attribute_map_inv = {'target_ip': 'source', 'target_port': 'src_port', 'opposite_ip': 'destination',
                                  'opposite_port': 'dst_port', 'duration': 'duration', 'target_pkts': 'in_pkts',
                                  'opposite_pkts': 'out_pkts', 'target_bytes': 'in_bytes', 'opposite_bytes': 'out_bytes',
                                  'start_time': 'first', 'end_time': 'last'}
        self.feature_list = ['target_ip', 'target_key_port', 'opposite_key_port', 'card_target_port', 'card_opposite_ip',
                             'card_opposite_port', 'sum_target_pkts', 'sum_opposite_pkts', 'sum_target_bytes',
                             'sum_opposite_bytes', 'sum_dur',
                        'avg_target_pkts', 'avg_opposite_pkts', 'avg_target_bytes', 'avg_opposite_bytes', 'avg_dur',
                        'std_target_pkts', 'std_opposite_pkts', 'std_target_bytes', 'std_opposite_bytes', 'std_dur',
                        'len_flows', 'pst_per_flows',
                        'etp_target_pkts', 'etp_opposite_pkts', 'etp_target_bytes', 'etp_opposite_bytes',
                        'in_bytes_per_pkt', 'in_pkts_per_byte', 'out_bytes_per_pkt', 'out_pkts_per_byte',
                        'in_out_bytes', 'in_out_pkts', 'out_in_bytes', 'out_in_pkts',
                        'in_bytes_per_sec', 'in_pkts_per_sec', 'out_bytes_per_sec', 'out_pkts_per_sec',
                        'etp_opposite_ip', 'etp_opposite_port', 'etp_target_port', 'etp_pattern',
                        'freq_target_port', 'freq_opposite_port', 'freq_opposite_ip',
                        'sum_int', 'avg_int', 'std_int', 'etp_int']

        self.feature_func_map = {
            'target_ip':
                lambda x: str(IPv4Address(int(IPv4Address(x.target_ip)) >> (32 - 32) << (32 - 32))),
            'target_key_port':
                lambda x: sorted(
                    zip(*np.unique(x['target_port'], return_counts=True)), key=lambda y: y[1], reverse=True)[0][0],
            'opposite_key_port':
                lambda x: sorted(
                    zip(*np.unique(x['opposite_port'], return_counts=True)), key=lambda y: y[1], reverse=True)[0][0],
            'card_target_port':
                lambda x: len(set(x['target_port'])),
            'card_opposite_ip':
                lambda x: len(set(x['opposite_ip'])),
            'card_opposite_port':
                lambda x: len(set(x['opposite_port'])),
            'sum_target_pkts':
                lambda x: np.sum(x['target_pkts']),
            'sum_opposite_pkts':
                lambda x: np.sum(x['opposite_pkts']),
            'sum_target_bytes':
                lambda x: np.sum(x['target_bytes']),
            'sum_opposite_bytes':
                lambda x: np.sum(x['opposite_bytes']),
            'sum_dur':
                lambda x: np.sum(x['duration']),
            'avg_target_pkts':
                lambda x: np.mean(x['target_pkts']),
            'avg_opposite_pkts':
                lambda x: np.mean(x['opposite_pkts']),
            'avg_target_bytes':
                lambda x: np.mean(x['target_bytes']),
            'avg_opposite_bytes':
                lambda x: np.mean(x['opposite_bytes']),
            'avg_dur':
                lambda x: np.mean(x['duration']),
            'std_target_pkts':
                lambda x: np.std(x['target_pkts']),
            'std_opposite_pkts':
                lambda x: np.std(x['opposite_pkts']),
            'std_target_bytes':
                lambda x: np.std(x['target_bytes']),
            'std_opposite_bytes':
                lambda x: np.std(x['opposite_bytes']),
            'std_dur':
                lambda x: np.std(x['duration']),
            'len_flows':
                lambda x: len(x['opposite_ip']),
            'pst_per_flows':
                lambda x: sorted(zip(*np.unique("{}_{}_{}_{}".format(x['target_pkts'], x['opposite_pkts'],
                                                                     x['target_bytes'], x['opposite_bytes']),
                                                return_counts=True)), reverse=True, key=lambda y: y[1])[0][1],
            'etp_target_pkts':
                lambda x: entropy(np.unique(x['target_pkts'], return_counts=True)[1]),
            'etp_opposite_pkts':
                lambda x: entropy(np.unique(x['opposite_pkts'], return_counts=True)[1]),
            'etp_target_bytes':
                lambda x: entropy(np.unique(x['target_bytes'], return_counts=True)[1]),
            'etp_opposite_bytes':
                lambda x: entropy(np.unique(x['opposite_bytes'], return_counts=True)[1]),
            'in_bytes_per_pkt':
                lambda x: np.sum(x['target_bytes']) / max(np.sum(x['target_pkts']), 0.1),
            'in_pkts_per_byte':
                lambda x: np.sum(x['target_pkts']) / max(np.sum(x['target_bytes']), 0.1),
            'out_bytes_per_pkt':
                lambda x: np.sum(x['opposite_bytes']) / max(np.sum(x['opposite_pkts']), 0.1),
            'out_pkts_per_byte':
                lambda x: np.sum(x['opposite_pkts']) / max(np.sum(x['opposite_bytes']), 0.1),
            'in_out_bytes':
                lambda x: np.sum(x['target_bytes']) / max(np.sum(x['opposite_bytes']), 0.1),
            'in_out_pkts':
                lambda x: np.sum(x['target_pkts']) / max(np.sum(x['opposite_pkts']), 0.1),
            'out_in_bytes':
                lambda x: np.sum(x['opposite_bytes']) / max(np.sum(x['target_bytes']), 0.1),
            'out_in_pkts':
                lambda x: np.sum(x['opposite_pkts']) / max(np.sum(x['target_pkts']), 0.1),
            'in_bytes_per_sec':
                lambda x: np.sum(x['target_bytes']) / max(np.sum(x['duration']), 0.1),
            'in_pkts_per_sec':
                lambda x: np.sum(x['target_pkts']) / max(np.sum(x['duration']), 0.1),
            'out_bytes_per_sec':
                lambda x: np.sum(x['opposite_bytes']) / max(np.sum(x['duration']), 0.1),
            'out_pkts_per_sec':
                lambda x: np.sum(x['opposite_pkts']) / max(np.sum(x['duration']), 0.1),
            'etp_opposite_ip':
                lambda x: entropy(np.unique(x['opposite_ip'], return_counts=True)[1]),
            'etp_opposite_port':
                lambda x: entropy(np.unique(x['opposite_port'], return_counts=True)[1]),
            'etp_target_port':
                lambda x: entropy(np.unique(x['target_port'], return_counts=True)[1]),
            'etp_pattern':
                lambda x: sorted(zip(*np.unique("{}_{}_{}_{}".format(x['target_pkts'], x['opposite_pkts'],
                                                                     x['target_bytes'], x['opposite_bytes']),
                                                return_counts=True)), reverse=True, key=lambda y: y[1])[0][1] / len(
                    x['target_pkts']),
            'freq_target_port':
                lambda x: sorted(
                    zip(*np.unique(x['target_port'], return_counts=True)), key=lambda y: y[1],
                    reverse=True)[0][0] / len(x['target_port']),
            'freq_opposite_port':
                lambda x: sorted(
                    zip(*np.unique(x['opposite_port'], return_counts=True)), key=lambda y: y[1],
                    reverse=True)[0][0] / len(x['opposite_port']),
            'freq_opposite_ip':
                lambda x: sorted(
                    zip(*np.unique(x['target_port'], return_counts=True)), key=lambda y: y[1],
                    reverse=True)[0][0] / len(x['opposite_ip']),
            'sum_int':
                lambda x: np.sum(get_int(x['start_time'], x['end_time'])),
            'avg_int':
                lambda x: np.mean(get_int(x['start_time'], x['end_time'])),
            'std_int':
                lambda x: np.std(get_int(x['start_time'], x['end_time'])),
            'etp_int':
                lambda x: entropy(get_int(x['start_time'], x['end_time'])) if np.sum(
                    get_int(x['start_time'], x['end_time'])) != 0 else 0
        }
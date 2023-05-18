import gc
import os

import optparse

from preprocess import preprocessing, load_raw
from result import save_result, save_config
from model import train, test
from utils import format_date, init_config


def experiments(config):

    # data_load
    print("Data Load...")
    train_path, test_path, inside_ip_set, score_dict = load_raw(config['label_set'])

    print("Data Preprocessing...")
    train_x, test_x, test_label, ip_list = preprocessing(train_path, test_path, inside_ip_set, score_dict, config)
    print("Model Training...")
    model = train(train_x)
    print("Inferencing")
    rce_list = test(test_x, model)

    print("Save Result")
    save_result(test_label, rce_list, ip_list, score_dict, config)

def main():
    parser = optparse.OptionParser('usage %prog -t timeout -m min_sample -l Abused')
    parser.add_option('--mode', dest='mode', type='string', help='Profiling Mode(Count/Time)')
    parser.add_option('--method', dest='method', type='string', help='Profiling Mode(discard/hybrid)')
    parser.add_option('-m', dest='min_sample', type='int', help='minimum sample count')
    parser.add_option('--hybrid', dest='hybrid_count', type='int', help='minimum sample theta')
    parser.add_option('-t', dest='timeout', type='int', help='maximum timeout')
    parser.add_option('-p', dest='data_path', type='string', help='pickle data path')
    parser.add_option('-l', dest='label_set', type='string', help='label dictionary(Abused/GN)')
    parser.add_option('--abused', dest='abused', type='int', help='threshold to devide normal and mal when using abused score ')
    parser.add_option('--pre', dest='preprocessing_path', type='string', help='preprocessed pickle path')
    parser.add_option('--topk', dest='k', type='int', help='top k-th reconstruction error rate')
    (options, args) = parser.parse_args()
    config = init_config(options)

    save_dir = r"D:\Behavior_Profiling_result"
    save_path = os.path.join(save_dir, format_date(6))
    os.mkdir(os.path.join(save_dir, format_date(6)))
    config['save_path'] = save_path
    # for timeout in [60*60, 4*60*60, 12*60*60, 48*60*60]:
    #     config['timeout'] = timeout
    #     experiments(config)
    #     gc.collect()

    experiments(config)
    save_config(config)


if __name__ == '__main__':
    main()
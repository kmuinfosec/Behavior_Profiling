import gc
import os

import optparse

from preprocess import preprocessing, load_raw
from result import save_result, save_config
from model import train, test
from utils import format_date


def experiments(min_sample, timeout, save_path, hybrid_count=3, preprocessing_path=False,label_set='Abused'):

    # data_load
    print("Data Load...")
    train_path, test_path, inside_ip_set, score_dict = load_raw(label_set)

    if preprocessing_path:
        train_path = rf"{preprocessing_path}/train_profile.pkl"
        test_path = rf"{preprocessing_path}/test_profile.pkl"

    print("Data Preprocessing...")
    train_x, test_x, test_label, ip_list = preprocessing(min_sample, timeout, train_path, test_path, save_path,
                                                         inside_ip_set, score_dict, hybrid_count)
    print("Model Training...")
    model = train(train_x)
    print("Inferencing")
    rce_list = test(test_x, model)

    print("Save Result")
    save_result(min_sample, timeout, save_path, test_label, rce_list, ip_list, score_dict, label_set)

def main():
    parser = optparse.OptionParser('usage %prog -t timeout -m min_sample -l Abused')
    parser.add_option('-m', dest='min_sample', type='int', help='minimum sample count')
    parser.add_option('--hybrid', dest='hybrid_count', type='int', help='minimum sample theta')
    parser.add_option('-t', dest='timeout', type='int', help='maximum timeout')
    parser.add_option('-p', dest='data_path', type='string', help='pickle data path')
    parser.add_option('-l', dest='label_set', type='string', help='label dictionary(Abused/GN)')
    parser.add_option('--pre', dest='preprocessing_path', type='string', help='preprocessed pickle path')
    (options, args) = parser.parse_args()
    min_sample = options.min_sample
    timeout = options.timeout
    label_set = options.label_set
    hybrid_count = options.hybrid_count
    if options.preprocessing_path:
        preprocessing_path = os.path.abspath(options.preprocessing_path)
    else:
        preprocessing_path = False

    save_dir = r"D:\Behavior_Profiling_result"
    save_path = os.path.join(save_dir, format_date(6))
    os.mkdir(os.path.join(save_dir, format_date(6)))
    # for timeout in [60, 4*60, 12*60, 48*60]:
    #     experiments(min_sample, timeout, label_set)
    #     gc.collect()

    experiments(min_sample, timeout, save_path, hybrid_count, preprocessing_path, label_set)
    save_config(save_path, min_sample, hybrid_count, timeout, preprocessing_path, label_set)


if __name__ == '__main__':
    main()
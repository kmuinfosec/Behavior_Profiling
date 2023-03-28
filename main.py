import gc

import optparse

from preprocess import preprocessing, load_raw
from result import save_result
from model import train, test


def experiments(min_sample, timeout, save_path, label_set='Abused'):

    # data_load
    print("Data Load...")
    train_path, test_path, inside_ip_set, score_dict = load_raw(label_set)
    print("Data Preprocessing...")
    train_x, test_x, test_label, ip_list = preprocessing(min_sample, timeout, train_path, test_path,
                                                         inside_ip_set, score_dict)
    print("Model Training...")
    model = train(train_x)
    print("Inferencing")
    rce_list = test(test_x, model)

    print("Save Result")
    save_result(min_sample, timeout, save_path, test_label, rce_list, ip_list, score_dict, label_set)

def main():
    parser = optparse.OptionParser('usage %prog -t timeout -m min_sample -l Abused')
    parser.add_option('-m', dest='min_sample', type='int', help='minimum sample count')
    parser.add_option('-t', dest='timeout', type='int', help='maximum timeout')
    parser.add_option('-p', dest='data_path', type='string', help='pickle data path')
    parser.add_option('-l', dest='label_set', type='string', help='label dictionary(Abused/GN)')
    (options, args) = parser.parse_args()
    min_sample = options.min_sample
    timeout = options.timeout
    label_set = options.label_set

    save_path = r"D:\behavior(over)_result"

    # for timeout in [60, 4*60, 12*60, 48*60]:
    #     experiments(min_sample, timeout, label_set)
    #     gc.collect()

    experiments(min_sample, timeout, save_path, label_set)


if __name__ == '__main__':
    main()
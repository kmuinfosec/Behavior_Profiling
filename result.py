import csv
import os.path
import pickle

import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, accuracy_score, classification_report


def reconstruct_rce(label, rce, ip_list, config):
    rce_dict = {}
    ret_label = []
    ret_rce = []
    ret_ip = []
    for idx, ip in enumerate(ip_list):
        if ip not in rce_dict:
            rce_dict[ip] = {'label': label[idx], 'rce': []}
        rce_dict[ip]['rce'].append(rce[idx])

    if config['output_method'] == 'min':
        for ip in rce_dict:
            ret_label.append(rce_dict[ip]['label'])
            ret_rce.append(min(rce_dict[ip]['rce']))
    elif config['output_method'] == 'max':
        for ip in rce_dict:
            ret_label.append(rce_dict[ip]['label'])
            ret_rce.append(max(rce_dict[ip]['rce']))
    elif config['output_method'] == 'mean':
        for ip in rce_dict:
            ret_label.append(rce_dict[ip]['label'])
            ret_rce.append(np.mean(rce_dict[ip]['rce']))
    return ret_label, ret_rce, ret_ip


def print_optim_f1(tmp_label, recon, config):
    print("Save Result...")
    tmp_label = [abs(i-1) for i in tmp_label]
    precisions, recalls, thresholds = precision_recall_curve(tmp_label, recon)
    target_f1 = 0
    target_idx = 0
    for idx in range(len(thresholds)):
        if (recalls[idx] + precisions[idx]) == 0:
            f1 = 0
        else :
            f1 = 2 * (recalls[idx] * precisions[idx]) / (recalls[idx] + precisions[idx])
        if target_f1 < f1:
            target_f1 = f1
            target_idx = idx
    tmp_pred = [1 if i >= thresholds[target_idx] else 0 for i in recon]
    acc = accuracy_score(tmp_label, tmp_pred)
    print("Accuracy:", acc)
    print("Recall:", recalls[target_idx])
    print("Precision:", precisions[target_idx])
    print("F1-Score:", target_f1)
    print("Threshold:", thresholds[target_idx])

    save_path = rf"{config['save_path']}\({config['label_set']})result_min_{config['min_sample']}_to_{config['timeout']}_hy_{config['hybrid_count']}.txt"
    if os.path.exists(save_path):
        save_path =  rf"{config['save_path']}\({config['label_set']})IPresult_min_{config['min_sample']}_to_{config['timeout']}_hy_{config['hybrid_count']}.txt"
    with open(save_path, 'w', encoding='utf-8', newline='') as f:
        f.write("Accuracy : " + str(acc) + "\n")
        f.write("Recall : " + str(recalls[target_idx]) + "\n")
        f.write("Precision: " + str(precisions[target_idx]) + "\n")
        f.write("F1-Score : " + str(target_f1) + "\n")
        f.write("Threshold :" + str(thresholds[target_idx]) + "\n")
        f.write(classification_report(tmp_label, tmp_pred))

    return thresholds[target_idx]


def print_plt(label, rce_list, th, config):
    print("Save plt...")

    attack_loss = []
    benign_loss = []
    for idx, rce in enumerate(rce_list):
        if label[idx] == 1:
            attack_loss.append(rce_list[idx])
        else:
            benign_loss.append(rce_list[idx])

    plt.figure(figsize=(9, 7))
    plt.rc('font', size=15)
    plt.hist(attack_loss, color='r', range=(0, 3), alpha=0.5, bins=100, log=True, label='Malicious IP')
    plt.hist(benign_loss, color='b', range=(0, 3), alpha=0.5, bins=100, log=True, label='Benign IP')
    plt.axvline(th, color='red')
    plt.legend()
    plt.ylabel("IP Count")
    plt.xlabel("Reconstruction Error")
    save_path = rf'{config["save_path"]}\({config["label_set"]})plt_min_{config["min_sample"]}_to_{config["timeout"]}_hy_{config["hybrid_count"]}.png'
    if os.path.exists(save_path):
        save_path = rf'{config["save_path"]}\({config["label_set"]})IPplt_min_{config["min_sample"]}_to_{config["timeout"]}_hy_{config["hybrid_count"]}.png'
    plt.savefig(save_path)


def print_csv(label, rce_list, ip_list, score_dict, config):
    print("Save csv...")
    sorted_rce_idx = np.argsort(rce_list)
    save_path = rf"{config['save_path']}\({config['label_set']})top{config['k']}_min_{config['min_sample']}_to_{config['timeout']}_hy_{config['hybrid_count']}.csv"
    if os.path.exists(save_path):
        save_path = rf"{config['save_path']}\({config['label_set']})IPtop{config['k']}_min_{config['min_sample']}_to_{config['timeout']}_hy_{config['hybrid_count']}.csv"
    with open(save_path, 'w', encoding='utf-8', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["IP", "RCE", "SCORE", "Label"])
        for idx, i in enumerate(sorted_rce_idx):
            if idx >= config['k']:
                break
            else:
                csv_writer.writerow([ip_list[i], rce_list[i],
                                     score_dict[ip_list[i]] if ip_list[i] in score_dict else 0, label[i]])


def save_result(label, rce_list, ip_list, score_dict, config):
    th = print_optim_f1(label, rce_list, config)
    print_plt(label, rce_list, th, config)
    print_csv(label, rce_list, ip_list, score_dict, config)
    if config['save_rce']:
        with open(rf"{config['save_path']}\rce_list.pkl", 'wb') as f:
            pickle.dump(rce_list, f)

    ip_label, ip_rce_list, ip_list = reconstruct_rce(label, rce_list, ip_list, config)
    th = print_optim_f1(ip_label, ip_rce_list, config)
    print_plt(ip_label, ip_rce_list, th, config)
    print_csv(ip_label, ip_rce_list, ip_list, score_dict, config)


def save_config(config):
    with open(rf"{config['save_path']}\config.txt", 'w', newline='', encoding='utf-8') as f:
        f.write(f'Profiling : {config["mode"]}\n')
        f.write(f'Method : {config["method"]}\n')
        f.write(f'MinSample : {config["min_sample"]}\n')
        f.write(f'Hybrid Count : {config["hybrid_count"]}\n')
        f.write(f'IP rce aggregate Method : {config["output_method"]}\n')
        f.write(f'Timeout : {config["timeout"]}\n')
        if config["preprocessing_path"]:
            f.write(f'Preprocessing Path : {config["preprocessing_path"]}\n')
        f.write(f'Label API : {config["label_set"]}\n')
        f.write(f'Save Path : {config["save_path"]}')


def main():
    pass


if __name__ == '__main__':
    main()
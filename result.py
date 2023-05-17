import csv

import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, accuracy_score, classification_report


def print_optim_f1(min_sample, timeout, save_path, tmp_label, recon, label_set):
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

    with open(rf"{save_path}\({label_set})result_min_{min_sample}_to_{timeout}.txt", 'w', encoding='utf-8', newline='') as f:
        f.write("Accuracy : " + str(acc) + "\n")
        f.write("Recall : " + str(recalls[target_idx]) + "\n")
        f.write("Precision: " + str(precisions[target_idx]) + "\n")
        f.write("F1-Score : " + str(target_f1) + "\n")
        f.write("Threshold :" + str(thresholds[target_idx]) + "\n")
        f.write(classification_report(tmp_label, tmp_pred))


def print_plt(min_sample, timeout, save_path, label, rce_list, label_set):
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
    plt.legend()
    plt.ylabel("IP Count")
    plt.xlabel("Reconstruction Error")
    plt.savefig(rf"{save_path}\({label_set})plt_min_{min_sample}_to_{timeout}.png")


def print_csv(min_sample, timeout, save_path, label, rce_list, ip_list, score_dict, label_set, n=100):
    print("Save csv...")
    sorted_rce_idx = np.argsort(rce_list)
    with open(rf"{save_path}\({label_set})top{n}_min_{min_sample}_to_{timeout}.csv", 'w', encoding='utf-8', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["IP", "RCE", "SCORE", "Label"])
        for idx, i in enumerate(sorted_rce_idx):
            if idx >= 100:
                break
            else:
                csv_writer.writerow([ip_list[i], rce_list[i],
                                     score_dict[ip_list[i]] if ip_list[i] in score_dict else 0, label[i]])


def save_result(min_sample, timeout, save_path, label, rce_list, ip_list, score_dict, label_set):
    print_optim_f1(min_sample, timeout, save_path, label, rce_list, label_set)
    print_plt(min_sample, timeout, save_path, label, rce_list, label_set)
    print_csv(min_sample, timeout, save_path, label, rce_list, ip_list, score_dict, label_set, 100)


def save_config(save_path, min_sample, hybrid_count, timeout, preprocessing_path, label_set):
    with open(rf"{save_path}\config.txt", 'w', newline='', encoding='utf-8') as f:
        f.write('Profiling : Count\n')
        f.write('Method : Hybrid\n')
        f.write(f'MinSample :{min_sample}\n')
        f.write(f'Hybrid Count :{hybrid_count}\n')
        f.write(f'Timeout :{timeout}\n')
        if preprocessing_path:
            f.write(f'Preprocessing Path :{preprocessing_path}\n')
        f.write(f'Label API :{label_set}\n')
        f.write(f'Save Path : {save_path}')


def main():
    pass


if __name__ == '__main__':
    main()
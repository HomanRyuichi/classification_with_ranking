import csv
from sklearn.metrics import classification_report, precision_score
import pandas as pd
import numpy as np



def make_precision_csv(output_folder, phase):
    with open(f'{output_folder}/{phase}/precision_{phase}.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'precision', 'precision_0', 'precision_1', 'precision_2', 'precision_3'])


def save_precision_result(epoch, output, output_folder, phase):
    true = output["class_true"]
    pred = output["class_pred_1hot"]
    target_names = ['mayo0', 'mayo1', 'mayo2', 'mayo3']

    precision = precision_score(true, pred, average='macro')
    report = classification_report(true, pred, labels=np.arange(len(target_names)), target_names=target_names, output_dict=True)

    precision0 = report['mayo0']['precision']
    precision1 = report['mayo1']['precision']
    precision2 = report['mayo2']['precision']
    precision3 = report['mayo3']['precision']

    with open(f'{output_folder}/{phase}/precision_{phase}.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch, precision, precision0, precision1, precision2, precision3])



def make_precision_avg_csv(fold, output_folder):
    with open(f'{output_folder}/precision_{fold}fold_avg.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['precision', 'precision_0', 'precision_1', 'precision_2', 'precision_3'])


def save_precision_avg(fold, output_folder):

    precisions = []
    precisions_0 = []
    precisions_1 = []
    precisions_2 = []
    precisions_3 = []

    for i in range(1, fold+1):
        csv_path = f"{output_folder}/fold{i}/test/precision_test.csv"
        precisions.append(pd.read_csv(csv_path, usecols=[1]))
        precisions_0.append(pd.read_csv(csv_path, usecols=[2]))
        precisions_1.append(pd.read_csv(csv_path, usecols=[3]))
        precisions_2.append(pd.read_csv(csv_path, usecols=[4]))
        precisions_3.append(pd.read_csv(csv_path, usecols=[5]))
    
    avg_precision = np.mean(precisions)
    avg_precision_0 = np.mean(precisions_0)
    avg_precision_1 = np.mean(precisions_1)
    avg_precision_2 = np.mean(precisions_2)
    avg_precision_3 = np.mean(precisions_3)

    with open(f'{output_folder}/precision_{fold}fold_avg.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([avg_precision, avg_precision_0, avg_precision_1, avg_precision_2, avg_precision_3])




import csv
from sklearn.metrics import classification_report, recall_score
import pandas as pd
import numpy as np


def make_recall_csv(output_folder, phase):
    with open(f'{output_folder}/{phase}/recall_{phase}.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'recall', 'recall_0', 'recall_1', 'recall_2', 'recall_3'])


def save_recall_result(epoch, output, output_folder, phase):
    true = output["class_true"]
    pred = output["class_pred_1hot"]
    target_names = ['mayo0', 'mayo1', 'mayo2', 'mayo3']

    recall = recall_score(true, pred, average='macro')
    report = classification_report(true, pred, labels=np.arange(len(target_names)), target_names=target_names, output_dict=True)

    recall0 = report['mayo0']['recall']
    recall1 = report['mayo1']['recall']
    recall2 = report['mayo2']['recall']
    recall3 = report['mayo3']['recall']

    with open(f'{output_folder}/{phase}/recall_{phase}.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch, recall, recall0, recall1, recall2, recall3])



def make_recall_avg_csv(fold, output_folder):
    with open(f'{output_folder}/recall_{fold}fold_avg.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['recall', 'recall_0', 'recall_1', 'recall_2', 'recall_3'])


def save_recall_avg(fold, output_folder):

    recalls = []
    recalls_0 = []
    recalls_1 = []
    recalls_2 = []
    recalls_3 = []

    for i in range(1, fold+1):
        csv_path = f"{output_folder}/fold{i}/test/recall_test.csv"
        recalls.append(pd.read_csv(csv_path, usecols=[1]))
        recalls_0.append(pd.read_csv(csv_path, usecols=[2]))
        recalls_1.append(pd.read_csv(csv_path, usecols=[3]))
        recalls_2.append(pd.read_csv(csv_path, usecols=[4]))
        recalls_3.append(pd.read_csv(csv_path, usecols=[5]))
    
    avg_recall = np.mean(recalls)
    avg_recall_0 = np.mean(recalls_0)
    avg_recall_1 = np.mean(recalls_1)
    avg_recall_2 = np.mean(recalls_2)
    avg_recall_3 = np.mean(recalls_3)

    with open(f'{output_folder}/recall_{fold}fold_avg.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([avg_recall, avg_recall_0, avg_recall_1, avg_recall_2, avg_recall_3])



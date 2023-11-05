import csv
from sklearn.metrics import classification_report, f1_score
import pandas as pd
import numpy as np

def make_f1score_csv(output_folder, phase):
    with open(f'{output_folder}/{phase}/f1score_{phase}.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'f1score', 'f1score_0', 'f1score_1', 'f1score_2', 'f1score_3'])


def save_f1score_result(epoch, output, output_folder, phase):
    true = output["class_true"]
    pred = output["class_pred_1hot"]
    target_names = ['mayo0', 'mayo1', 'mayo2', 'mayo3']

    f1score = f1_score(true, pred, average='macro')
    report = classification_report(true, pred, labels=np.arange(len(target_names)), target_names=target_names, output_dict=True)

    f1score0 = report['mayo0']['f1-score']
    f1score1 = report['mayo1']['f1-score']
    f1score2 = report['mayo2']['f1-score']
    f1score3 = report['mayo3']['f1-score']

    with open(f'{output_folder}/{phase}/f1score_{phase}.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch, f1score, f1score0, f1score1, f1score2, f1score3])



def make_f1score_avg_csv(fold, output_folder):
    with open(f'{output_folder}/f1score_{fold}fold_avg.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['f1score', 'f1score_0', 'f1score_1', 'f1score_2', 'f1score_3'])


def save_f1score_avg(fold, output_folder):

    f1scores = []
    f1scores_0 = []
    f1scores_1 = []
    f1scores_2 = []
    f1scores_3 = []

    for i in range(1, fold+1):
        csv_path = f"{output_folder}/fold{i}/test/f1score_test.csv"
        f1scores.append(pd.read_csv(csv_path, usecols=[1]))
        f1scores_0.append(pd.read_csv(csv_path, usecols=[2]))
        f1scores_1.append(pd.read_csv(csv_path, usecols=[3]))
        f1scores_2.append(pd.read_csv(csv_path, usecols=[4]))
        f1scores_3.append(pd.read_csv(csv_path, usecols=[5]))
    
    avg_f1score = np.mean(f1scores)
    avg_f1score_0 = np.mean(f1scores_0)
    avg_f1score_1 = np.mean(f1scores_1)
    avg_f1score_2 = np.mean(f1scores_2)
    avg_f1score_3 = np.mean(f1scores_3)

    with open(f'{output_folder}/f1score_{fold}fold_avg.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([avg_f1score, avg_f1score_0, avg_f1score_1, avg_f1score_2, avg_f1score_3])
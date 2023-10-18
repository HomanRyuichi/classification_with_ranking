import csv
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np



def make_accuracy_csv(output_folder, phase):
    with open(f'{output_folder}/{phase}/accuracy_{phase}.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'accuracy', 'accuracy_0', 'accuracy_1', 'accuracy_2', 'accuracy_3'])


def save_accuracy_result(epoch, output, output_folder, phase):
    true = output["class_true"]
    pred = output["class_pred_1hot"]
    classes = ['mayo0', 'mayo1', 'mayo2', 'mayo3']

    accuracy = accuracy_score(true, pred)
    report = classification_report(true, pred, target_names=classes, output_dict=True)

    # accuracy0 = report['mayo0']['accuracy']
    # accuracy1 = report['mayo1']['accuracy']
    # accuracy2 = report['mayo2']['accuracy']
    # accuracy3 = report['mayo3']['accuracy']

    with open(f'{output_folder}/{phase}/accuracy_{phase}.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch, accuracy])


def make_accuracy_graph(output_folder):

    train_csv_path = f"{output_folder}/train/accuracy_train.csv"
    val_csv_path = f"{output_folder}/val/accuracy_val.csv"

    epoch = pd.read_csv(train_csv_path, usecols=[0])
    train_accuracy = pd.read_csv(train_csv_path, usecols=[1])
    val_accuracy = pd.read_csv(val_csv_path, usecols=[1])
    
    plt.clf()
    fig = plt.figure()
    plt.plot(epoch, train_accuracy, label='train')
    plt.plot(epoch, val_accuracy, label='val')
    plt.legend()      
    plt.grid()        
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title(f"accuracy history")
    plt_name = f"{output_folder}/accuracy.jpg"
    fig.savefig(plt_name)
    plt.close(fig)


def make_accuracy_avg_csv(fold, output_folder):
    with open(f'{output_folder}/accuracy_{fold}fold_avg.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['accuracy', 'accuracy_0', 'accuracy_1', 'accuracy_2', 'accuracy_3'])


def save_accuracy_avg(fold, output_folder):

    accuracys = []
    accuracys_0 = []
    accuracys_1 = []
    accuracys_2 = []
    accuracys_3 = []

    for i in range(1, fold+1):
        csv_path = f"{output_folder}/fold{i}/test/accuracy_test.csv"
        accuracy = pd.read_csv(csv_path, usecols=[1])
        accuracys.append(accuracy)
    
    avg_accuracy = np.mean(accuracys)

    with open(f'{output_folder}/accuracy_{fold}fold_avg.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([avg_accuracy])





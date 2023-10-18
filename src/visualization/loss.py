import csv
import pandas as pd
from matplotlib import pyplot as plt


def make_loss_csv(output_folder, phase):
    with open(f'{output_folder}/{phase}/loss_{phase}.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'loss', 'class_loss', 'ranking_loss'])


def save_loss_result(epoch, output, output_folder, phase):
    loss = output["loss"]
    rank_loss = output["rank_loss"]
    class_loss = output["loss"]

    with open(f'{output_folder}/{phase}/loss_{phase}.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch, loss, rank_loss, class_loss])


def make_loss_graph(output_folder):
    cols = ["all", "rank", "class"]

    for i, col in enumerate(cols):
        train_csv_path = f"{output_folder}/train/loss_train.csv"
        val_csv_path = f"{output_folder}/val/loss_val.csv"

        epoch = pd.read_csv(train_csv_path, usecols=[0])
        train_loss = pd.read_csv(train_csv_path, usecols=[i+1])
        val_loss = pd.read_csv(val_csv_path, usecols=[i+1])
        
        plt.clf()
        fig = plt.figure()
        plt.plot(epoch, train_loss, label='train')
        plt.plot(epoch, val_loss, label='val')
        plt.legend()      
        plt.grid()        
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title(f"{col} loss history")
        plt_name = f"{output_folder}/loss_{col}.jpg"
        fig.savefig(plt_name)
        plt.close(fig)


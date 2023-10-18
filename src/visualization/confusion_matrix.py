import pandas as pd
import seaborn as sn 
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

def make_confusion_matrix(epoch, true, pred, output_folder, phase):
    cm = confusion_matrix(true, pred)
    classes = [0, 1, 2, 3]
    df_cm = pd.DataFrame(cm, index = [i for i in classes], columns = [i for i in classes])
    
    plt.clf()
    fig = plt.figure(figsize = (12,7))
    sn.set(font_scale=1.5)
    plt.xlabel("Pred")
    plt.ylabel("GT")
    plt.title(f'{phase} confusion matrix')
    sn.heatmap(df_cm, annot=True, fmt='d')
    plt_name = f"{output_folder}/{phase}/cm_{phase}_epoch{epoch}.jpg"    
    fig.savefig(plt_name)
    mpl.style.use('default')
    plt.close(fig)



def make_confusion_matrix_ratio(epoch, true, pred, output_folder, phase):
    cm = confusion_matrix(true, pred)
    classes = [0, 1, 2, 3]
    df_cm = pd.DataFrame(cm / np.sum(cm, axis=1)[:, None], index = [i for i in classes],
                    columns = [i for i in classes])
    plt.clf()
    fig = plt.figure(figsize = (12,7))
    sn.set(font_scale=1.5)
    plt.xlabel("Pred")
    plt.ylabel("GT")
    plt.title(f'{phase} confusion matrix')
    sn.heatmap(df_cm, annot=True, fmt='.2f')
    plt_name = f"{output_folder}/{phase}/cm_ratio_{phase}_epoch{epoch}.jpg"
    fig.savefig(plt_name)
    mpl.style.use('default')
    plt.close(fig)

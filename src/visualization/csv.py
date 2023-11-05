import csv
from visualization.loss import make_loss_csv, save_loss_result
from visualization.accuracy import make_accuracy_csv, save_accuracy_result, make_accuracy_avg_csv, save_accuracy_avg
from visualization.precision import make_precision_csv, save_precision_result, save_precision_avg, make_precision_avg_csv
from visualization.recall import make_recall_csv, save_recall_result, make_recall_avg_csv, save_recall_avg
from visualization.f1score import make_f1score_csv, save_f1score_result, make_f1score_avg_csv, save_f1score_avg



def make_all_csv(output_folder):
    phases = ["train", "val", "test"]
    for phase in phases:
        make_loss_csv(output_folder, phase)
        make_accuracy_csv(output_folder, phase)
        make_precision_csv(output_folder, phase)
        make_recall_csv(output_folder, phase)
        make_f1score_csv(output_folder, phase)


def save_epoch_result(epoch, output, output_folder, phase):
    if phase != "test":
        save_loss_result(epoch, output, output_folder, phase)
    save_accuracy_result(epoch, output, output_folder, phase)
    save_precision_result(epoch, output, output_folder, phase)
    save_recall_result(epoch, output, output_folder, phase)
    save_f1score_result(epoch, output, output_folder, phase)



def make_avg_csv(fold, output_folder):
    make_accuracy_avg_csv(fold, output_folder)
    make_precision_avg_csv(fold, output_folder)
    make_recall_avg_csv(fold, output_folder)
    make_f1score_avg_csv(fold, output_folder)
    
    


def save_avg_result(fold, output_folder):
    save_accuracy_avg(fold, output_folder)
    save_precision_avg(fold, output_folder)
    save_recall_avg(fold, output_folder)
    save_f1score_avg(fold, output_folder)







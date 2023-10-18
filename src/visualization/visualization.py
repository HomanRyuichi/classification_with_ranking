from visualization.loss import make_loss_graph
from visualization.accuracy import make_accuracy_graph
from visualization.confusion_matrix import make_confusion_matrix, make_confusion_matrix_ratio


def visualize_train_results(best_epoch, train_best_output, val_best_output, output_folder):
    make_loss_graph(output_folder)
    make_accuracy_graph(output_folder)
    make_confusion_matrix(best_epoch, train_best_output["class_true"], train_best_output["class_pred_1hot"], output_folder, phase="train")
    make_confusion_matrix(best_epoch, val_best_output["class_true"], val_best_output["class_pred_1hot"], output_folder, phase="val")
    make_confusion_matrix_ratio(best_epoch, train_best_output["class_true"], train_best_output["class_pred_1hot"], output_folder, phase="train")
    make_confusion_matrix_ratio(best_epoch, val_best_output["class_true"], val_best_output["class_pred_1hot"], output_folder, phase="val")


def visualize_test_results(best_epoch, test_output, output_folder):
    make_confusion_matrix(best_epoch, test_output["class_true"], test_output["class_pred_1hot"], output_folder, phase="test")
    make_confusion_matrix_ratio(best_epoch, test_output["class_true"], test_output["class_pred_1hot"], output_folder, phase="test")




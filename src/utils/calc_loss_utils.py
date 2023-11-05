import torch

def calc_thresholds(c1):
    _, indices = torch.max(c1, dim=1, keepdim=True)
    thresholds = indices

    return thresholds


def sum_thresholded_pred(preds, thresholds):
    thresholded_preds = torch.zeros(preds.size(0), 2)

    for i in range(preds.size(0)):
        threshold = thresholds[i]
        thresholded_preds[i, 0] = preds[i, :threshold+1].sum()
        thresholded_preds[i, 1] = preds[i, threshold+1:].sum()

    return thresholded_preds


def calc_2cls_CEloss(thresholded_preds, labels):
    print("a")

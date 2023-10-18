import torch
from tqdm import tqdm
import numpy as np

@torch.no_grad()
def feature_multitask_rank_class(net, AL_loader, device):
    gts = []
    features = []

    net.eval()

    for images, labels in tqdm(AL_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        feature = net(images)

        labels = labels.data.cpu().numpy()
        feature = feature.data.cpu().numpy().tolist()

        for i in range(len(labels)):
            gts.append(labels[i])
            features.append(feature[i])

    return np.array(gts), np.array(features)
import torch
from tqdm import tqdm
import numpy as np


def train_classifier(mode, net, AL_loader, optimizer, class_criterion, device):
    log_loss = 0
    class_true = []
    class_pred = []
    class_pred_to_1cls = []
    datasize = 0

    if mode ==  'train':
        net.train()
    elif mode == 'val' or mode == 'test':
        net.eval()

    for images, labels in tqdm(AL_loader):
        images = images.to(device)
        labels = labels.to(device)

        c = net(images)
        c = c.squeeze()

        # exception
        if list(labels.shape) == [1]:
            c = c.reshape(1, 4)
            
        loss = class_criterion(c, labels)

        if mode ==  'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        labels = labels.data.cpu().numpy()
        c = c.data.cpu().numpy()

        for i in range(len(labels)):
            class_true.append(labels[i])
            class_pred.append(c[i])
            class_pred_to_1cls.append(np.argmax(c[i]))
        
        log_loss += loss.item() * c.size
        datasize += c.size
    
    return log_loss/datasize, class_true, class_pred, class_pred_to_1cls



@torch.no_grad()
def test_classifier(net, AL_loader, class_criterion, device):
    log_class_loss = 0
    class_true = []
    class_pred = []
    class_pred_to_1cls = []
    data_size = 0
    
    net.eval()

    for images, labels in tqdm(AL_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        c = net(images)
        c = c.squeeze()

        class_loss = class_criterion(c, labels)

        labels = labels.data.cpu().numpy()
        c = c.data.cpu().numpy()

        for i in range(len(labels)):
            class_true.append(labels[i])
            class_pred.append(c[i])
            class_pred_to_1cls.append(np.argmax(c[i]))

        log_class_loss += class_loss.item() * c.size
        data_size += c.size

    return log_class_loss/data_size, class_true, class_pred, class_pred_to_1cls


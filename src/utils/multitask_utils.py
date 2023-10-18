import torch
from tqdm import tqdm
import numpy as np


#rank-class
def train_multitask(net, AL_loader, RL_loader, optimizer, rank_criterion, class_criterion, device, epoch, phase):
    log_loss = 0
    log_rank_loss = 0
    log_class_loss = 0
    rank_grad = []
    class_grad = [[] for _ in range(4)]
    rank_true = []
    rank_pred = []
    class_true = []
    class_pred = []
    class_pred_1hot = []
    path = []
    s_size = 0
    c_size = 0

    if phase ==  'train':
        net.train()
    elif phase == 'val' or phase == 'test':
        net.eval()
        
    for (images, labels), (image0s, image1s, ranks, pair_mayos, pair_paths) in (zip(tqdm(AL_loader), tqdm(RL_loader))):
        images = images.to(device)
        labels = labels.to(device)
        image0s = image0s.to(device)
        image1s = image1s.to(device)
        ranks = ranks.to(device)

        s, c = net(images)
        s0, c0 = net(image0s)
        s1, c1 = net(image1s)

        s0 = s0.squeeze()
        s1 = s1.squeeze()
        c = c.squeeze()

        # exception
        if list(labels.shape) == [1]:
            c = c.reshape(1, 4)
        elif list(s0.shape) == []:
            s0 = s0.reshape(1)
            s1 = s1.reshape(1)

        rank_loss = rank_criterion(s0,s1,ranks)
        class_loss = class_criterion(c, labels)
        loss = rank_loss + class_loss

        if phase ==  'train':
            optimizer.zero_grad()
            loss.backward()
            # save grad
            rank_grad_array = net.module.fc_task1.weight.grad.squeeze()
            rank_grad_array = rank_grad_array.data.cpu().numpy()
            rank_grad.append(sum([abs(i) for i in rank_grad_array]) / len(rank_grad_array))
            class_grad_array = net.module.fc_task2.weight.grad
            class_grad_array = class_grad_array.data.cpu().numpy()
            for j in range(len(class_grad_array)):
                class_grad[j].append(sum([abs(i) for i in class_grad_array[j]]) / len(class_grad_array[j]))
            optimizer.step()
        
        ranks = ranks.data.cpu().numpy()
        s0 = s0.data.cpu().numpy()
        s1 = s1.data.cpu().numpy()
        labels = labels.data.cpu().numpy()
        c = c.data.cpu().numpy()

        #動いたあとここをもっと省略して書きたい extend
        #pathいらん
        for i in range(len(ranks)):
            rank_true.append([pair_mayos[0][i], pair_mayos[1][i]])
            rank_pred.append([s0[i], s1[i]])
            path.append([pair_paths[0][i], pair_paths[1][i]])

        for i in range(len(labels)):
            class_true.append(labels[i])
            class_pred.append(c[i])
            class_pred_1hot.append(np.argmax(c[i]))
        
        log_loss += loss.item() * (s0.size + c.size)
        log_rank_loss += rank_loss.item() * s0.size
        log_class_loss += class_loss.item() * c.size
        s_size += s0.size
        c_size += c.size
    
    # calc grad
    grad = {'rank_grad':0, 'class_grad':0, 'class_grad0':0, 'class_grad1':0, 'class_grad2':0, 'class_grad3':0}
    if phase == 'train':
        grad['rank_grad'] = sum([abs(i) for i in rank_grad]) / len(rank_grad)
        class_grad_tmp = 0
        tmp = 0
        for j in range(len(class_grad_array)):
                grad[f'class_grad{j}'] = sum([abs(i) for i in class_grad_array[j]]) / len(class_grad_array[j])
                class_grad_tmp += sum([abs(i) for i in class_grad_array[j]]) / len(class_grad_array[j])
                tmp += 1
        grad['class_grad'] = class_grad_tmp / tmp

    output = {
        "loss": log_loss/(s_size+c_size),
        "rank_loss": log_rank_loss/s_size,
        "class_loss": log_class_loss/c_size,
        "rank_true": rank_true,
        "rank_pred": rank_pred,
        "class_true": class_true,
        "class_pred": class_pred,
        "class_pred_1hot": class_pred_1hot,
        "grad": grad
    }
    
    return output



@torch.no_grad()
def test_multitask(net, AL_loader, device):
    class_true = []
    class_pred = []
    class_pred_1hot = []

    net.eval()

    for images, labels in tqdm(AL_loader):
        images = images.to(device)
        labels = labels.to(device)

        s, c = net(images)
        c = c.squeeze()

        labels = labels.data.cpu().numpy()
        c = c.data.cpu().numpy()

        for i in range(len(labels)):
            class_true.append(labels[i])
            class_pred.append(c[i])
            class_pred_1hot.append(np.argmax(c[i]))

    output = {
        "class_true":  class_true,
        "class_pred": class_pred,
        "class_pred_1hot": class_pred_1hot
    }

    return output









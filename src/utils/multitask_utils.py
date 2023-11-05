import torch
from tqdm import tqdm
import numpy as np
from utils.calc_loss_utils import calc_thresholds, sum_thresholded_pred


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
    r_size = 0
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

        c = net(images)
        r0 = net(image0s)   # high
        r1 = net(image1s)   # low

        r0 = r0.squeeze()
        r1 = r1.squeeze()
        c = c.squeeze()

        # exception
        if list(labels.shape) == [1]:
            c = c.reshape(1, 4)
        elif list(r0.shape) == []:
            r0 = r0.reshape(1)
            r1 = r1.reshape(1)

        
        # rank_loss = rank_criterion(r0,r1,ranks)
        # loss = rank_loss + class_loss
        # print("a")

        thresholds = calc_thresholds(r1)
        thresholded_r0 = sum_thresholded_pred(r0, thresholds)
        thresholded_r1 = sum_thresholded_pred(r1, thresholds)
        target_high = torch.ones(len(r0), dtype=torch.long) 
        target_low = torch.zeros(len(r1), dtype=torch.long) 
        rank_loss_high = class_criterion(thresholded_r0, target_high)
        rank_loss_low = class_criterion(thresholded_r1, target_low)
        rank_loss = (rank_loss_high + rank_loss_low) / 2

        class_loss = class_criterion(c, labels)
        loss = rank_loss + class_loss

        if phase ==  'train':
            optimizer.zero_grad()
            loss.backward()
            # save grad
            # rank_grad_array = net.module.fc_task1.weight.grad.squeeze()
            # rank_grad_array = rank_grad_array.data.cpu().numpy()
            # rank_grad.append(sum([abs(i) for i in rank_grad_array]) / len(rank_grad_array))
            # class_grad_array = net.module.fc_task2.weight.grad
            # class_grad_array = class_grad_array.data.cpu().numpy()
            # for j in range(len(class_grad_array)):
            #     class_grad[j].append(sum([abs(i) for i in class_grad_array[j]]) / len(class_grad_array[j]))
            optimizer.step()
        
        ranks = ranks.data.cpu().numpy()
        r0 = r0.data.cpu().numpy()
        r1 = r1.data.cpu().numpy()
        labels = labels.data.cpu().numpy()
        c = c.data.cpu().numpy()

        for i in range(len(ranks)):
            rank_true.append([pair_mayos[0][i], pair_mayos[1][i]])
            rank_pred.append([r0[i], r1[i]])
            path.append([pair_paths[0][i], pair_paths[1][i]])

        for i in range(len(labels)):
            class_true.append(labels[i])
            class_pred.append(c[i])
            class_pred_1hot.append(np.argmax(c[i]))
        
        log_loss += loss.item() * (r0.size + c.size)
        log_rank_loss += rank_loss.item() * r0.size
        log_rank_loss_high = rank_loss_high.item() * r0.size
        log_rank_loss_low = rank_loss_low.item() * r0.size
        log_class_loss += class_loss.item() * c.size
        r_size += r0.size
        c_size += c.size
    
    # calc grad
    # grad = {'rank_grad':0, 'class_grad':0, 'class_grad0':0, 'class_grad1':0, 'class_grad2':0, 'class_grad3':0}
    # if phase == 'train':
    #     grad['rank_grad'] = sum([abs(i) for i in rank_grad]) / len(rank_grad)
    #     class_grad_tmp = 0
    #     tmp = 0
    #     for j in range(len(class_grad_array)):
    #             grad[f'class_grad{j}'] = sum([abs(i) for i in class_grad_array[j]]) / len(class_grad_array[j])
    #             class_grad_tmp += sum([abs(i) for i in class_grad_array[j]]) / len(class_grad_array[j])
    #             tmp += 1
    #     grad['class_grad'] = class_grad_tmp / tmp

    output = {
        "loss": log_loss/(r_size+c_size),
        "rank_loss": log_rank_loss/r_size,
        "class_loss": log_class_loss/c_size,
        "rank_true": rank_true,
        "rank_pred": rank_pred,
        "class_true": class_true,
        "class_pred": class_pred,
        "class_pred_1hot": class_pred_1hot,
        # "grad": grad
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

        c = net(images)
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









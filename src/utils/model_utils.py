import numpy as np
import torch
import torch.nn as nn
from torchvision.models import vgg19, resnet50, resnet18, densenet169, densenet121

    
class DenseNet169WithTwoOutputs(nn.Module):
    def __init__(self, num_classes_task1, num_classes_task2):
        super(DenseNet169WithTwoOutputs, self).__init__()
        self.densenet = densenet169(pretrained=True)
        num_features = self.densenet.classifier.in_features

        self.densenet.classifier = nn.Identity()


        # タスク1用の全結合層を追加
        self.fc_task1 = nn.Linear(num_features, num_classes_task1)

        # タスク2用の全結合層を追加
        self.fc_task2 = nn.Linear(num_features, num_classes_task2)
    
    def forward(self, x):
        features = self.densenet(x)
        out = torch.nn.functional.relu(features, inplace=True)
        output_task1 = self.fc_task1(out)
        output_task2 = self.fc_task2(out)
        return output_task1, output_task2
    

class DenseNet169WithOneOutputs(nn.Module):
    def __init__(self, num_classes):
        super(DenseNet169WithOneOutputs, self).__init__()
        self.densenet = densenet169(pretrained=True)
        num_features = self.densenet.classifier.in_features

        self.densenet.classifier = nn.Identity()

        self.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        features = self.densenet(x)
        out = torch.nn.functional.relu(features, inplace=True)
        output = self.fc(out)
        return output




def classifier_model(name='densenet', pretrained=True,  n_class=4, **kwargs):
    if name == 'densenet':
        model = build_densenet169(pretrained=pretrained, n_class=n_class)
    elif name == 'resnet':
        model = build_resnet50(pretrained=pretrained, n_class=n_class)
    elif name == 'resnet18':
        model = build_resnet18(pretrained=pretrained, n_class=n_class)
    elif name == 'vgg':
        model = build_vgg19(pretrained=pretrained, n_class=n_class)
    return model


def regression_model(name='densenet', pretrained=True, **kwargs):
    if name == 'densenet':
        model = build_densenet169(pretrained=pretrained)
    elif name == 'resnet':
        model = build_resnet50(pretrained=pretrained)
    elif name == 'vgg':
        model = build_vgg19(pretrained=pretrained)
    return model



def build_densenet169(pretrained=True, n_class=1):
    model = densenet169(pretrained=pretrained)
    in_features = model.classifier.in_features 
    model.classifier = nn.Sequential(
        nn.Linear(in_features=in_features, out_features=n_class),
        Lambda(lambda x: x.squeeze())
    )
    return model


def build_resnet50(pretrained=True, n_class=1):
    model = resnet50(pretrained=pretrained)
    in_features = model.fc.in_features 
    model.fc = nn.Sequential(
        nn.Linear(in_features=in_features, out_features=n_class),
        Lambda(lambda x: x.squeeze())
    )
    return model

def build_resnet18(pretrained=True, n_class=1):
    model = resnet18(pretrained=pretrained)
    in_features = model.fc.in_features 
    model.fc = nn.Sequential(
        nn.Linear(in_features=in_features, out_features=n_class),
        Lambda(lambda x: x.squeeze())
    )
    return model


def build_vgg19(pretrained=True, n_class=1):
    model = vgg19(pretrained=pretrained)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Sequential(
        nn.Linear(in_features=in_features, out_features=n_class),
        Lambda(lambda x: x.squeeze())
    )
    return model


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


def ranknet_loss(s0, s1, t):
    o = torch.sigmoid(s0 - s1)
    loss = (-t * o.log() - (1 - t) * (1 - o).log()).mean()
    return loss


def score2label(scores):
    labels = []
    
    for score in scores:
        if score < 0.5:
            labels.append(0)
        
        elif score >= 0.5 and score < 1.5:
            labels.append(1)

        elif score >= 1.5 and score <2.5:
            labels.append(2)
        
        elif score >= 2.5:
            labels.append(3)

    return np.array(labels)


def load_weight(net, path):
    module_dict = torch.load(path)
    net.load_state_dict(module_dict)
    return net




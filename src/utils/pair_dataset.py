import argparse
import os
import glob
import shutil
import itertools
import random
from PIL import Image
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import Dataset

from utils.utils import fix_seed

def make_pair_dataset(in_dir: str, out_dir: str, n_pairs: int):
    fix_seed(seed=777)
    img = []
    label = []

    for mayo in range(4):
        img_path = glob.glob(f'{in_dir}/Mayo{mayo}/*')
        img += img_path
        label += [mayo]*len(img_path)
    idx_list = [i for i in range(len(label))]
    all_pairs = list(itertools.combinations(idx_list, 2))
    idx_pairs_list = [i for i in range(len(all_pairs))]
    idx_pairs = random.sample(idx_pairs_list, n_pairs)
    count = 0

    for idx in idx_pairs:
        img1, img2 = img[all_pairs[idx][0]], img[all_pairs[idx][1]]
        label1, label2 = label[all_pairs[idx][0]], label[all_pairs[idx][1]]
        if label1 > label2:
            os.makedirs(f'{out_dir}/HL/{count}/H')
            os.makedirs(f'{out_dir}/HL/{count}/L')
            shutil.copy(img1, f'{out_dir}/HL/{count}/H')
            shutil.copy(img2, f'{out_dir}/HL/{count}/L')
        elif label2 > label1:
            os.makedirs(f'{out_dir}/HL/{count}/H')
            os.makedirs(f'{out_dir}/HL/{count}/L')
            shutil.copy(img1, f'{out_dir}/HL/{count}/L')
            shutil.copy(img2, f'{out_dir}/HL/{count}/H')
        elif label1 == label2:
            os.makedirs(f'{out_dir}/S/{count}')
            shutil.copy(img1, f'{out_dir}/S/{count}')
            shutil.copy(img2, f'{out_dir}/S/{count}')
        else:
            print('err')
        
        count += 1

class AL_Dataset(Dataset):
    def __init__(self, root, transforms=None) -> None:
        super().__init__()
        self.img_path = []
        self.label = []
        for mayo in range(4):
            path = glob.glob(root + f'/Mayo{mayo}/**')
            self.img_path += path
            self.label += [torch.tensor(mayo)]*len(path)
        
        self.transforms = transforms
    
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, index):
        image = Image.open(self.img_path[index])
        label = self.label[index]
        if self.transforms:
            image = self.transforms(image)
        return image, label

#2クラスで試す時用のデータセット
class AL_Dataset2(Dataset):
    def __init__(self, root, transforms=None) -> None:
        super().__init__()
        self.img_path = []
        self.label = []
        for i, mayo in enumerate([0,3]):
            path = glob.glob(root + f'/Mayo{mayo}/**')
            self.img_path += path
            self.label += [torch.tensor(i)]*len(path)
        
        self.transforms = transforms
    
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, index):
        image = Image.open(self.img_path[index])
        label = self.label[index]
        if self.transforms:
            image = self.transforms(image)
        return image, label


class RL_Dataset(Dataset):
    def __init__(self, root, transforms=None) -> None:
        super().__init__()
        self.pair_path = []
        self.rank = []
        self.pair_mayo = []

        hl_path = glob.glob(root+f'/HL/**')
        for path in hl_path:
            h_path = glob.glob(path + '/H/**')[0]
            l_path = glob.glob(path + '/L/**')[0]
            self.pair_path.append([h_path, l_path])
            self.rank.append(torch.tensor(1.0))
            h_name = os.path.basename(h_path)
            l_name = os.path.basename(l_path)
            h_label = h_name.split("_")[0][-1]
            l_label = l_name.split("_")[0][-1]
            self.pair_mayo.append([int(h_label), int(l_label)])
            
        s_path = glob.glob(root+f'/S/**')
        for path in s_path:
            s1_path = glob.glob(path + '/**')[0]
            s2_path = glob.glob(path + '/**')[1]
            self.pair_path.append(glob.glob(path + '/**'))
            self.rank.append(torch.tensor(0.5))
            s1_name = os.path.basename(s1_path)
            s2_name = os.path.basename(s2_path)
            s1_label = s1_name.split("_")[0][-1]
            s2_label = s2_name.split("_")[0][-1]
            self.pair_mayo.append([int(s1_label), int(s2_label)])
        
        self.transforms = transforms
    
    def __len__(self):
        return len(self.rank)
    
    def __getitem__(self, index):
        image0 = Image.open(self.pair_path[index][0])
        image1 = Image.open(self.pair_path[index][1])
        rank = self.rank[index]
        pair_mayo = self.pair_mayo[index]
        pair_path = self.pair_path[index]
        if self.transforms:
            image0 = self.transforms(image0)
            image1 = self.transforms(image1)
        return image0, image1, rank, pair_mayo, pair_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YAML file')
    parser.add_argument('--in_dir', type=str, default='./dataset/5fold/fold1/train-val')
    parser.add_argument('--out_dir', type=str, default='./dataset/5fold-RL/fold1/train-val')
    parser.add_argument('--n_pairs', type=int, default=5000)
    args = parser.parse_args()
    make_pair_dataset(in_dir=args.in_dir, out_dir=args.out_dir, n_pairs=args.n_pairs)
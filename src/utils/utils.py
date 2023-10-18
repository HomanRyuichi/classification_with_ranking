import os
from datetime import datetime
import random

import numpy as np
import torch
import torch.optim as optim

def get_date():
    now = datetime.now()
    date = now.strftime('%Y%m%d_%H%M%S')
    return date

def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def worker_init_fn(seed):
    random.seed(seed)
    np.random.seed(seed)

def make_optimizer(params, name, **kwargs):
    # Optimizer
    return optim.__dict__[name](params, **kwargs)

class EarlyStopping:
    """earlystoppingクラス"""

    def __init__(self, patience=20, mode='loss', path='checkpoint', verbose=True, **kwargs):
        """引数最小値の非更新数カウンタ、表示設定、モデル格納path"""

        self.patience = patience    #設定ストップカウンタ
        self.verbose = verbose      #表示の有無
        self.counter = 0            #現在のカウンタ値
        self.best_score = None      #ベストスコア
        self.early_stop = False     #ストップフラグ

        self.mode = mode
        if mode == 'loss':
            self.val_score_min = np.Inf   #前回のベストスコア記憶
        elif self.mode == 'F1':
            self.val_score_max = 0. 
        self.path = path             #ベストモデル格納path

    def __call__(self, val_score, net, optim):
        """
        特殊(call)メソッド
        実際に学習ループ内で最小lossを更新したか否かを計算させる部分
        """

        if self.mode == 'loss':
            score = -val_score
        elif self.mode == 'score':
            score = val_score

        if self.best_score is None:  #1Epoch目の処理
            self.best_score = score   #1Epoch目はそのままベストスコアとして記録する
            self.checkpoint(val_score, net, optim)  #記録後にモデルを保存してスコア表示する
        elif score < self.best_score:  # ベストスコアを更新できなかった場合
            self.counter += 1   #ストップカウンタを+1
            if self.verbose:  #表示を有効にした場合は経過を表示
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')  #現在のカウンタを表示する 
            if self.counter >= self.patience:  #設定カウントを上回ったらストップフラグをTrueに変更
                self.early_stop = True
        else:  #ベストスコアを更新した場合
            self.best_score = score  #ベストスコアを上書き
            self.checkpoint(val_score, net, optim)  #モデルを保存してスコア表示
            self.counter = 0  #ストップカウンタリセット

    def checkpoint(self, val_score, net, optim):
        '''ベストスコア更新時に実行されるチェックポイント関数'''
        if self.verbose:  #表示を有効にした場合は、前回のベストスコアからどれだけ更新したか？を表示
            if self.mode == 'loss':
                print(f'Validation loss decreased ({self.val_score_min:.6f} --> {val_score:.6f}).  Saving model ...')
                self.val_score_min = val_score  #その時のlossを記録する
            elif self.mode == 'score':
                print(f'Validation score increased ({self.val_score_max:.6f} --> {val_score:.6f}).  Saving model ...')
                self.val_score_max = val_score  #その時のF1を記録する
        torch.save(net.state_dict(), f'{self.path}/net.ckpt')  #ベストモデルを指定したpathに保存
        torch.save(optim.state_dict(), f'{self.path}/optim.ckpt')  #ベストモデルを指定したpathに保存
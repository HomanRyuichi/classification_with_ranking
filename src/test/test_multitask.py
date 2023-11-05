import os
import argparse
import yaml
import shutil

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from sklearn.metrics import f1_score

from init_path import init_path
init_path()
from utils.pair_dataset import AL_Dataset, RL_Dataset
from utils.utils import fix_seed, get_date, worker_init_fn, make_optimizer, EarlyStopping
from utils.multitask_utils import train_multitask, test_multitask
from utils.model_utils import ranknet_loss, DenseNet169WithTwoOutputs
from visualization.csv import make_all_csv, save_epoch_result, make_avg_csv, save_avg_result
from visualization.visualization import visualize_train_results, visualize_test_results




def main(args):
    # Initial settings
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    output_folder = f'{args.output_folder}/fold{args.fold}'
    # os.makedirs(f'output_{date}', exist_ok=True)
    os.makedirs(f'{output_folder}', exist_ok=True)
    os.makedirs(f'{output_folder}/test', exist_ok=True)
    make_all_csv(output_folder)

    RL_rate = config['RL_rate']
    os.makedirs(f'models/RL{RL_rate}%', exist_ok=True)

    
    fix_seed(seed=config['seed'])

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    

    # Configure data loader
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # transforms.Normalize((0.5,), (0.5,)), if you don't use pretrain model
    ])

    dataset_path = config['dataset']
    AL_dataset_path =  f"{dataset_path}/RL{RL_rate}%/5fold-copy"
    RL_dataset_path =  f"{dataset_path}/RL{RL_rate}%"

    
    test_AL_dataset = AL_Dataset(root=AL_dataset_path + f'/fold{args.fold}/test', transforms=train_transform)
    
    test_AL_loader = DataLoader(test_AL_dataset, 
                              batch_size=config['AL_data_loader']['batch_size'],
                              shuffle=False,
                              num_workers=config['AL_data_loader']['num_workers'],
                              pin_memory=config['AL_data_loader']['pin_memory'],
                              drop_last=config['AL_data_loader']['drop_last'],
                              worker_init_fn=worker_init_fn(seed=config['seed'])
                              )
    
    # Build net
    num_classes_task1 = 1
    num_classes_task2 = 4
    model_path = f"./models/RL{RL_rate}%/best_model_fold{args.fold}_*epoch.pt"
    net = DenseNet169WithTwoOutputs(num_classes_task1, num_classes_task2)
    net = nn.DataParallel(net)
    net.load_state_dict(torch.load(model_path))
    net = net.to(device)

    
    # test
    test_output = test_multitask(
                            net=best_model,
                            AL_loader=test_AL_loader,
                            device=device
                            )
    
    save_epoch_result(best_epoch, test_output, output_folder, phase="test")

    
    # visualize
    visualize_test_results(best_epoch, test_output, output_folder)

    # save log
    log_path = shutil.copy(args.config_path, output_folder)
    date = get_date()
    expr_log = {
        "date": date,
        "last_epoch": epoch,
        "best_epoch": best_epoch,
        "best_model_path": best_model_path,
        "last_model_path": last_model_path
    }
    with open(log_path, 'a')as f:
        yaml.dump(expr_log, f, default_flow_style=False, allow_unicode=True)



        
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YAML file')
    parser.add_argument('--config_path', type=str, default='./config/config_test.yaml', help='(.yaml)')
    parser.add_argument('--fold', type=int, default=1)

    date = get_date()
    output_folder = f'output_{date}'
    parser.add_argument('--output_folder', type=str, default=output_folder)

    # hold-out
    # args = parser.parse_args()
    # main(args)

    # 5-fold cross validation
    fold = 5
    for i in range(1, fold+1):
            args = parser.parse_args()
            args.fold = i
            main(args)
    make_avg_csv(fold, output_folder)
    save_avg_result(fold, output_folder)
    print(output_folder)
        

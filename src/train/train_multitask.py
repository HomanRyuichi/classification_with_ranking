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
    os.makedirs(f'{output_folder}/train', exist_ok=True)
    os.makedirs(f'{output_folder}/val', exist_ok=True)
    os.makedirs(f'{output_folder}/test', exist_ok=True)
    make_all_csv(output_folder)

    AL_rate = config['AL_rate']
    os.makedirs(f'models/AL{AL_rate}%/fold{args.fold}', exist_ok=True)

    
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
    AL_dataset_path =  f"{dataset_path}/AL{AL_rate}%/5fold-copy"
    RL_dataset_path =  f"{dataset_path}/AL{AL_rate}%"

    train_AL_dataset = AL_Dataset(root=AL_dataset_path + f'/fold{args.fold}/train', transforms=train_transform)
    val_AL_dataset = AL_Dataset(root=AL_dataset_path + f'/fold{args.fold}/val', transforms=train_transform)
    test_AL_dataset = AL_Dataset(root=AL_dataset_path + f'/fold{args.fold}/test', transforms=train_transform)
    train_RL_dataset = RL_Dataset(root=RL_dataset_path + f'/fold{args.fold}/train', transforms=train_transform)
    val_RL_dataset = RL_Dataset(root=RL_dataset_path + f'/fold{args.fold}/val', transforms=train_transform)
    
    train_AL_loader = DataLoader(train_AL_dataset, 
                              batch_size=config['AL_data_loader']['batch_size'],
                              shuffle=config['AL_data_loader']['shuffle'],
                              num_workers=config['AL_data_loader']['num_workers'],
                              pin_memory=config['AL_data_loader']['pin_memory'],
                              drop_last=config['AL_data_loader']['drop_last'],
                              worker_init_fn=worker_init_fn(seed=config['seed'])
                              )
    train_RL_loader = DataLoader(train_RL_dataset, 
                              batch_size=config['RL_data_loader']['batch_size'],
                              shuffle=config['RL_data_loader']['shuffle'],
                              num_workers=config['RL_data_loader']['num_workers'],
                              pin_memory=config['RL_data_loader']['pin_memory'],
                              drop_last=config['RL_data_loader']['drop_last'],
                              worker_init_fn=worker_init_fn(seed=config['seed'])
                              )
    val_AL_loader = DataLoader(val_AL_dataset, 
                              batch_size=config['AL_data_loader']['batch_size'],
                              shuffle=False,
                              num_workers=config['AL_data_loader']['num_workers'],
                              pin_memory=config['AL_data_loader']['pin_memory'],
                              drop_last=config['AL_data_loader']['drop_last'],
                              worker_init_fn=worker_init_fn(seed=config['seed'])
                              )
    val_RL_loader = DataLoader(val_RL_dataset, 
                              batch_size=config['RL_data_loader']['batch_size'],
                              shuffle=False,
                              num_workers=config['RL_data_loader']['num_workers'],
                              pin_memory=config['RL_data_loader']['pin_memory'],
                              drop_last=config['RL_data_loader']['drop_last'],
                              worker_init_fn=worker_init_fn(seed=config['seed'])
                              )
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
    net = DenseNet169WithTwoOutputs(num_classes_task1, num_classes_task2)
    net = nn.DataParallel(net)
    net = net.to(device)


    # Optimizers
    optimizer = make_optimizer(net.parameters(), **config['optimizer'])


    # Criterion
    rankloss = ranknet_loss
    CEloss = nn.CrossEntropyLoss()


    # Training
    best_f1score = 0
    best_epoch = 0
    best_train_output = 0
    best_val_output = 0

    for epoch in range(config['train_config']['epoch']):
        train_output = train_multitask(
                                        net=net,
                                        AL_loader=train_AL_loader,
                                        RL_loader=train_RL_loader,
                                        optimizer=optimizer,
                                        rank_criterion=rankloss,
                                        class_criterion=CEloss,
                                        device=device,
                                        epoch=epoch,
                                        phase='train'
                                        )

        val_output = train_multitask(
                            net=net,
                            AL_loader=val_AL_loader,
                            RL_loader=val_RL_loader,
                            optimizer=optimizer,
                            rank_criterion=rankloss,
                            class_criterion=CEloss,
                            device=device,
                            epoch=epoch,
                            phase='val'
                            )
        
        save_epoch_result(epoch, train_output, output_folder, phase="train")
        save_epoch_result(epoch, val_output, output_folder, phase="val")

        # es(val_loss, net, optimizer)
        # if es.early_stop:
        #     print("Early Stopping!")
        #     break

        # record best model
        f1score = f1_score(val_output["class_true"], val_output["class_pred_1hot"], average='macro')
        if best_f1score < f1score:
            best_model = net
            best_epoch = epoch
            best_train_output = train_output
            best_val_output = val_output
            best_f1score = f1score

        log = f'Fold{args.fold} [Epoch {epoch+1}/{config["train_config"]["epoch"]}]'
        log += ' ' + f'[train_loss: {train_output["loss"]:.4f}]'
        log += ' ' + f'[val_loss: {val_output["loss"]:.4f}]'
        print(log)


    # save model
    model_path = f"./models/AL{AL_rate}%/fold{args.fold}"
    best_model_path = f"{model_path}/best_model_fold{args.fold}.pt"
    last_model_path = f"{model_path}/last_model_fold{args.fold}.pt"
    torch.save(best_model.state_dict(), best_model_path)
    torch.save(net.state_dict(), last_model_path)

    
    # test
    test_output = test_multitask(
                            net=best_model,
                            AL_loader=test_AL_loader,
                            device=device
                            )
    
    save_epoch_result(best_epoch, test_output, output_folder, phase="test")

    
    # visualize
    visualize_train_results(best_epoch, best_train_output, best_val_output, output_folder)
    visualize_test_results(best_epoch, test_output, output_folder)


    # save log
    log_path = shutil.copy(args.config_path, output_folder)
    model_log_path = shutil.copy(args.config_path, model_path)
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
    with open(model_log_path, 'a')as f:
        yaml.dump(expr_log, f, default_flow_style=False, allow_unicode=True)



        
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YAML file')
    parser.add_argument('--config_path', type=str, default='./config/config_train.yaml', help='(.yaml)')
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

        

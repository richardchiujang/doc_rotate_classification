# -*- coding: utf-8 -*-
# @Time    : 2019/8/23 22:00
# @Author  : zhoujun

from __future__ import print_function
# import torch
# import deepspeed
import argparse
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['MASTER_ADDR'] = "localhost"
# os.environ['MASTER_PORT'] = "23456"
# os.environ['RANK'] = '0'
# os.environ['WORLD_SIZE'] = '2'
# os.environ['LOCAL_RANK'] = '0'
# os.environ['local_rank'] = '0'
# torch.distributed.init_process_group(backend="gloo")
# os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"

import anyconfig
import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder

def init_args():
    parser = argparse.ArgumentParser(description='FPN_classification.pytorch')
    parser.add_argument('--config_file', default='config/icdar2015_resnet18_FPN_DBhead_polyLR.yaml', type=str)
    parser.add_argument('--local_rank', dest='local_rank', default=0, type=int, help='Use distributed training')

    args = parser.parse_args()
    return args


def main(config):
    import torch
    from models import build_model  #, build_loss
    # from data_loader import get_dataloader
    from trainer import Trainer     
    # from post_processing import get_post_processing
    from utils import get_metric
    if torch.cuda.device_count() > 1:
        torch.cuda.set_device(args.local_rank)
        # torch.distributed.init_process_group(backend="nccl", init_method="env://", world_size=torch.cuda.device_count(), rank=args.local_rank)
        torch.distributed.init_process_group(backend="gloo", init_method="env://", world_size=torch.cuda.device_count(), rank=args.local_rank)
        config['distributed'] = True
    else:
        config['distributed'] = False
    config['local_rank'] = args.local_rank
    # config['local_rank'] = False 

    #train and test data directory
    data_dir = "datasets/train"
    test_data_dir = "datasets/test"

    #load the train and valid data
    dataset = ImageFolder(data_dir,transform = transforms.Compose([
        transforms.Resize((224,224)),transforms.ToTensor()
    ]))
    test_dataset = ImageFolder(test_data_dir,transforms.Compose([
        transforms.Resize((224,224)),transforms.ToTensor()
    ]))

    img, label = dataset[0]
    print(img.shape,label)    
    print("Follwing classes are there : \n",dataset.classes)


    from torch.utils.data.dataloader import DataLoader
    from torch.utils.data import random_split

    batch_size = 6 
    val_size = 1000
    train_size = len(dataset) - val_size 

    train_data,val_data = random_split(dataset,[train_size,val_size])
    print(f"Length of Train Data : {len(train_data)}")
    print(f"Length of Validation Data : {len(val_data)}")

    #output
    #Length of Train Data : 12034
    #Length of Validation Data : 2000

    #load the train and validation into batches.
    train_dl = DataLoader(train_data, batch_size, shuffle = True, num_workers = 4, pin_memory = False)
    val_dl = DataLoader(val_data, batch_size*2, num_workers = 4, pin_memory = False)


    # train_loader = get_dataloader(config['dataset']['train'], config['distributed'])
    train_loader = train_dl
    assert train_loader is not None
    if 'validate' in config['dataset']:
        # validate_loader = get_dataloader(config['dataset']['validate'], False)
        validate_loader = val_dl
    else:
        validate_loader = None

    criterion = torch.nn.CrossEntropyLoss() # .cuda()

    config['arch']['backbone']['in_channels'] = 3 if config['dataset']['train']['dataset']['args']['img_mode'] != 'GRAY' else 1
    model = build_model(config['arch'])

    # post_p = get_post_processing(config['post_processing'])
    metric = get_metric(config['metric'])

    trainer = Trainer(config=config,
                      model=model,
                      criterion=criterion,
                      train_loader=train_loader,
                    #   post_process=post_p,
                      metric_cls=metric,
                      validate_loader=validate_loader)
    trainer.train()


if __name__ == '__main__':
    import sys
    import pathlib
    __dir__ = pathlib.Path(os.path.abspath(__file__))
    sys.path.append(str(__dir__))
    sys.path.append(str(__dir__.parent.parent))
    # project = 'DBNet.pytorch'  # 工作项目根目录
    # sys.path.append(os.getcwd().split(project)[0] + project)

    from utils import parse_config

    args = init_args()
    assert os.path.exists(args.config_file)
    config = anyconfig.load(open(args.config_file, 'rb'))
    if 'base' in config:
        config = parse_config(config)
    print(config)
    main(config)

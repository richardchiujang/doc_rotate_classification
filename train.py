from __future__ import print_function
import argparse
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import anyconfig
import numpy as np
# import torch
# import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder


def init_args():
    parser = argparse.ArgumentParser(description='FPN_classification.pytorch')
    parser.add_argument('--config_file', default='config/resnet18_FPN_Classhead.yaml', type=str)
    args = parser.parse_args()
    return args

def main(config):
    import torch
    from drcmodels import build_model

    #train and test data directory
    train_data_dir = "datasets/train"
    valid_data_dir = "datasets/valid"
    test_data_dir = "datasets/test" 

    #load the train and valid data
    train_dataset = ImageFolder(train_data_dir,transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),     # 转换成形状为[C,H,W]，取值范围是[0,1.0]的torch.FloadTensor
    ]))
    valid_dataset = ImageFolder(valid_data_dir,transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),     # 转换成形状为[C,H,W]，取值范围是[0,1.0]的torch.FloadTensor
    ]))
    test_dataset = ImageFolder(test_data_dir,transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),     # 转换成形状为[C,H,W]，取值范围是[0,1.0]的torch.FloadTensor
    ]))

    from torch.utils.data.dataloader import DataLoader
    # from torch.utils.data import random_split

    batch_size = 256
    # val_size = 
    # train_size = len(dataset) - val_size 
    # train_data,val_data = random_split(dataset,[train_size,val_size])
    train_loader = DataLoader(train_dataset,batch_size,shuffle = True,num_workers = 10,pin_memory = True)
    validate_loader = DataLoader(valid_dataset,batch_size*2,num_workers = 10,pin_memory = True) 
    test_loader = DataLoader(test_dataset,batch_size*2,num_workers = 10,pin_memory = True) 
    print(f"Length of Train Data : {len(train_dataset)}")
    print(f"Length of Validation Data : {len(valid_dataset)}")
    print(f"Length of Test Data : {len(test_dataset)}")


    #load the train and validation into batches.
    # train_dl = DataLoader(train_data, batch_size, shuffle = True, num_workers = 10, pin_memory = True)
    # val_dl = DataLoader(val_data, batch_size*2, num_workers = 10, pin_memory = True)

    # train_loader = get_dataloader(config['dataset']['train'], config['distributed'])
    # train_loader = train_loader
    assert train_loader is not None
    # validate_loader = validate_loader
    assert validate_loader is not None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = torch.nn.CrossEntropyLoss() # .cuda()

    config['arch']['backbone']['in_channels'] = 3 # if config['dataset']['train']['dataset']['args']['img_mode'] != 'GRAY' else 1
    model = build_model(config['arch'])
    # checkpoint = torch.load('model.pth')
    # model.load_state_dict(checkpoint)
    # print('load model.pth')
    model = model.to(device)

    # print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    # model.train()
    train_loss_list = []
    train_acc_list = []
    validate_loss_list = []
    validate_acc_list = []    
    best_acc = 0.0
    for epoch in range(1, config['trainer']['epochs'] + 1):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device) # target = tensor([1, 2, 0, 1, 0, 1], device='cuda:0')
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)  # tensor(0.0567, device='cuda:0', grad_fn=<NllLossBackward0>)
            acc = (output.argmax(dim=1) == target).float().mean() # tensor(1., device='cuda:0')
            loss.backward()
            optimizer.step()

            if batch_idx % 30 == 0:
                train_loss_list.append(loss.item())
                train_acc_list.append(acc.item())
                batch = batch_idx * len(data)
                data_count = len(train_loader.dataset)
                percentage = (100. * batch_idx / len(train_loader))
                print(f'Epoch {epoch}: [{batch:5d} / {data_count}] ({percentage:.0f} %)' +
                    f'  Loss: {loss.item():.6f}' + f'  Accuracy: {acc.item():.6f}')

        model.eval()
        with torch.no_grad():
            epoch_loss = []
            epoch_acc = []
            for batch_idx, (data, target) in enumerate(validate_loader):
                data, target = data.to(device), target.to(device) # target = tensor([1, 2, 0, 1, 0, 1], device='cuda:0')
                output = model(data)
                loss = criterion(output, target)  # tensor(0.0567, device='cuda:0', grad_fn=<NllLossBackward0>)
                acc = (output.argmax(dim=1) == target).float().mean() # tensor(1., device='cuda:0')
                validate_loss_list.append(loss.item())
                validate_acc_list.append(acc.item())
                epoch_loss.append(loss.item())
                epoch_acc.append(acc.item())
            if np.mean(epoch_acc) > best_acc:
                best_acc = np.mean(epoch_acc)
                print('save model.pth', np.mean(epoch_loss), np.mean(epoch_acc))
                torch.save(model.state_dict(), 'model.pth')
        print(f'validation:   Loss: {np.mean(epoch_loss):.6f}' + f'  Accuracy: {np.mean(epoch_acc):.6f}')

    # model test phase
    model.eval()
    with torch.no_grad():
        epoch_loss = []
        epoch_acc = []
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device) # target = tensor([1, 2, 0, 1, 0, 1], device='cuda:0')
            output = model(data)
            loss = criterion(output, target)  # tensor(0.0567, device='cuda:0', grad_fn=<NllLossBackward0>)
            acc = (output.argmax(dim=1) == target).float().mean() # tensor(1., device='cuda:0')
            epoch_loss.append(loss.item())
            epoch_acc.append(acc.item())
    print(f'test phase:   Loss: {np.mean(epoch_loss):.6f}' + f'  Accuracy: {np.mean(epoch_acc):.6f}')





if __name__ == '__main__':
    import sys
    import pathlib
    __dir__ = pathlib.Path(os.path.abspath(__file__))
    sys.path.append(str(__dir__))
    sys.path.append(str(__dir__.parent.parent))
    from utils import parse_config
    args = init_args()
    assert os.path.exists(args.config_file)
    config = anyconfig.load(open(args.config_file, 'rb'))
    if 'base' in config:
        config = parse_config(config)
    # print(config)
    main(config)
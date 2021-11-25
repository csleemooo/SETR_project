import torch
from torch import nn
from torch.utils.data import DataLoader
import os

from setr.SETR import *
from setr.unet import UNet

from utils.data_load import sync_transform, oct_dataset
from utils.parse_args import parse_args
from utils.functions import *
from utils import functions

from torchmetrics.functional import iou
from torchvision import transforms
import pdb


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

colormap = {'IRF': [128, 0, 0],
            'SRF': [0, 128, 0],
            'PED': [0, 0, 128]}

class_index = {0: 'IRF', 1: 'SRF', 2: 'PED'}
num_of_class = colormap.__len__()

if __name__ == '__main__':

    args = parse_args()

    #     'SETR_Naive_S', 'SETR_Naive_L', 'SETR_Naive_H',
    #     'SETR_PUP_S', 'SETR_PUP_L', 'SETR_PUP_H',
    #     'SETR_MLA_S', 'SETR_MLA_L', 'SETR_MLA_H',

    used_model = eval(args.model)
    try : 
        aux_layers, model = used_model(dataset='oct')
        aux=True 
    except : 
        model = used_model(dataset='oct')
        aux=False 

    model = model.to(device)

    # data size: (650, 512), (496, 512), (885, 512), (1024, 512)
    dataset_train = oct_dataset(data_path=os.path.join(args.data_path, 'train', 'images'),
                                label_path=os.path.join(args.data_path, 'train', 'labels'),
                                sync_transform=sync_transform)

    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)

    transform_val = transforms.Compose([transforms.CenterCrop(480)])
    dataset_val = oct_dataset(data_path=os.path.join(args.data_path, 'test', 'images'),
                              label_path=os.path.join(args.data_path, 'test', 'labels'),
                              sync_transform=None, transform=transform_val)

    val_loader = DataLoader(dataset_val, batch_size=args.batch_size)

    if args.loss_weight != 1 : 
        criterion = functions.weighted_BCE(args.loss_weight)
    else : 
        criterion = nn.BCELoss()
        
    activation = torch.nn.Sigmoid()

    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, betas=(0.5, 0.99))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs//3, gamma=0.1)
    loss_train_set = []
    i_train_set = []
    u_train_set = []
    loss_val_set = []
    i_val_set = []
    u_val_set = []

    best_val_loss = 1e+4

    for epoch in range(args.epochs):

        model.train()
        loss_train_set.append([])
        i_train_set.append(0)
        u_train_set.append(0)

        for b_idx, [image, target, name] in enumerate(iter(train_loader)):

            image = image.to(device).float()
            target = target.to(device).float()

            optimizer.zero_grad()
            if aux : 
                pred = model(image, aux_layers)
            else : 
                pred = model(image)
            pred = activation(pred)

            loss = criterion(pred, target)

            loss_train_set[-1].append(loss.item())
            #iou_train_set[-1].append(iou(convert_to_label(pred), convert_to_label(target)).item())
            intersection, union, IoU, mIoU = functions.get_IoU(pred, target) 
            i_train_set[-1] += intersection
            u_train_set[-1] += union 
            
            
            loss.backward()
            optimizer.step()

            if (b_idx + 1) % args.ckpt == 0:  # checkpoint
                current_IoU = i_train_set[-1]/u_train_set[-1]
                print(pred[0,:,0])
                print("[Epoch: %d/%d iteration:%d/%d] Train Loss: %2.4f" % (
                    epoch + 1, args.epochs, b_idx + 1, len(train_loader), np.mean(loss_train_set[-1])), 
                      f'IoU : {current_IoU.tolist()}, mIoU : {current_IoU.mean().item()}'
                      )
                if args.fast_pass : 
                    break 
                
        scheduler.step()
        model.eval()
        save_set = [62, 194, 443, 3125, 3276, 3400, 5781, 5881]
        loss_val_set.append([])

        i_val_set.append(0)
        u_val_set.append(0)    
        with torch.no_grad() : 
            for b_idx, [image, target, name] in enumerate(iter(val_loader)):
    
                image = image.to(device).float()
                target = target.to(device).float()
    
                if aux : 
                    pred = model(image, aux_layers)
                else : 
                    pred = model(image)
                
                pred = activation(pred)
    
                loss = criterion(pred, target)
    
                loss_val_set[-1].append(loss.item())
                intersection, union, IoU, mIoU = functions.get_IoU(pred, target) 
                i_val_set[-1] += intersection
                u_val_set[-1] += union 
    
                for img_idx, img_num in enumerate([int(i.split('.')[0]) for i in name if int(i.split('.')[0]) in save_set]):
                    image_, target_, pred_ = image[img_idx][0].cpu().detach().numpy(), target[img_idx], pred[img_idx]
                    target_ = convert_to_RGB_image(convert_to_label(target_))
                    pred_ = convert_to_RGB_image(pred_)
    
                    save_image(image_, os.path.join(args.ckpt_path, 'test_', str(epoch)+'_'+str(img_num) + '_input.png'))
                    save_image(target_, os.path.join(args.ckpt_path, 'test_', str(epoch)+'_'+str(img_num) + '_target.png'))
                    save_image(pred_, os.path.join(args.ckpt_path, 'test_', str(epoch)+'_'+str(img_num) + '_pred.png' ''))
                    if args.fast_pass : 
                        break 
                
        current_IoU = i_val_set[-1]/u_val_set[-1]
        print("[Epoch: %d/%d] Validation Loss: %2.4f" % (
                epoch + 1, args.epochs, np.mean(loss_val_set[-1])), 
              f'IoU : {current_IoU.tolist()}, mIoU : {current_IoU.mean().item()}'
              )

        saved_data = {'epoch': epoch, 'model_state_dict': model.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict(),
                      'train_loss': loss_train_set, 'valid_loss': loss_val_set,
                      'train_IOU': (i_train_set, u_train_set), 'valid_IOU': (i_val_set, u_val_set)}
        torch.save(saved_data, os.path.join(args.ckpt_path, "last_model.pth"))

        if best_val_loss > np.mean(loss_val_set[-1]):
            torch.save(saved_data, os.path.join(args.ckpt_path, "best_model.pth"))
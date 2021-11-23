import torch
from torch import nn
from torch.utils.data import DataLoader
import os

from setr.SETR import *

from utils.data_load import sync_transform, oct_dataset
from utils.parse_args import parse_args
from utils.functions import *

from torchmetrics.functional import iou
from torchvision import transforms


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
    aux_layers, model = used_model(dataset='oct')

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


    criterion = nn.BCELoss()
    activation = torch.nn.Sigmoid()

    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, betas=(0.5, 0.99))

    loss_train_set = []
    iou_train_set = []
    loss_val_set = []
    iou_val_set = []

    best_val_loss = 1e+4

    for epoch in range(args.epochs):

        model.train()
        loss_train_set.append([])
        iou_train_set.append([])

        for b_idx, [image, target, name] in enumerate(iter(train_loader)):

            image = image.to(device).float()
            target = target.to(device).float()

            optimizer.zero_grad()
            pred = model(image, aux_layers)
            pred = activation(pred)

            loss = criterion(pred, target)

            loss_train_set[-1].append(loss.item())
            iou_train_set[-1].append(iou(convert_to_label(pred), convert_to_label(target)).item())

            loss.backward()
            optimizer.step()

            if (b_idx + 1) % args.ckpt == 0:  # checkpoint
                print("[Epoch: %d/%d iteration:%d] Train Loss: %2.4f, mIOU: %2.4f" % (
                    epoch + 1, args.epochs, b_idx + 1, np.mean(loss_train_set[-1]), np.mean(iou_train_set[-1])))

        model.eval()
        save_set = [62, 194, 443, 3125, 3276, 3400, 5781, 5881]
        loss_val_set.append([])
        iou_val_set.append([])
        for b_idx, [image, target, name] in enumerate(iter(val_loader)):

            image = image.to(device).float()
            target = target.to(device).float()

            pred = model(image, aux_layers)
            pred = activation(pred)

            loss = criterion(pred, target)

            loss_val_set[-1].append(loss.item())
            iou_val_set[-1].append(iou(convert_to_label(pred), convert_to_label(target)).item())

            for img_idx, img_num in enumerate([int(i.split('.')[0]) for i in name if int(i.split('.')[0]) in save_set]):
                image, target, pred = image[img_idx][0].cpu().detach().numpy(), target[img_idx], pred[img_idx]
                target = convert_to_RGB_image(convert_to_label(target))
                pred = convert_to_RGB_image(pred)

                save_image(image, os.path.join(args.ckpt_path, 'test_', str(img_num) + '_input.png'))
                save_image(target, os.path.join(args.ckpt_path, 'test_', str(img_num) + '_target.png'))
                save_image(pred, os.path.join(args.ckpt_path, 'test_', str(img_num) + '_pred.png' ''))

        else:
            print("[Epoch: %d/%d] Validation Loss: %2.4f, mIOU: %2.4f" % (
                epoch + 1, args.epochs, np.mean(loss_val_set[-1]), np.mean(iou_val_set[-1])))

        saved_data = {'epoch': epoch, 'model_state_dict': model.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict(),
                      'train_loss': loss_train_set, 'valid_loss': loss_val_set,
                      'train_IOU': iou_train_set, 'valid_IOU': iou_val_set}
        torch.save(saved_data, os.path.join(args.ckpt_path, "last_model.pth"))

        if best_val_loss > np.mean(loss_val_set[-1]):
            torch.save(saved_data, os.path.join(args.ckpt_path, "best_model.pth"))
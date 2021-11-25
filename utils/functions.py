import numpy as np
import matplotlib.pyplot as plt
import torch 

def convert_to_label(pred, confident=0.5):
    return (pred >confident) * 1

def get_IoU(pred, target, confident=0.5) : 
    B,C,H,W = pred.shape 
    pred = (pred.cpu()>confident).numpy() # boolean array (B,C,H,W)
    pred_background = np.logical_not(np.any(pred, axis=1, keepdims=True)) # (B,1,H,W)
    pred = np.concatenate([pred_background, pred], axis=1) # (B,C+1,H,W)
    
    target = (target.cpu()>0.5).numpy()
    target_background = np.logical_not(np.any(target, axis=1, keepdims=True))
    target = np.concatenate([target_background, target], axis=1)
    
    
    intersection = np.logical_and(pred, target) # (B,C+1,H,W)
    union = np.logical_or(pred, target)
    
    intersection = intersection.astype(int).sum(axis=(0,2,3))
    union = union.astype(int).sum(axis=(0,2,3))
    IoU = intersection/np.clip(union, a_min=1, a_max=None)
    mIoU = np.mean(IoU)
    
    return intersection, union, IoU, mIoU


def convert_to_RGB_image(img):

    img = img.cpu().detach().numpy()
    c, Nx, Ny = np.shape(img)
    RGB_img = np.zeros([Nx, Ny, 3], dtype=np.int)

    for c in range(3):
        #RGB_img[:, :, c] = (img[c, :, :] == 1).astype(np.uint8) * 128
        RGB_img[:, :, c] = (img[c, :, :]*128).astype(np.uint8)

    return RGB_img


def save_image(img, path):

    fig = plt.figure(1, figsize=[6, 6])
    if np.ndim(img) == 2:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(img)

    plt.axis('off')
    fig.savefig(path)
    plt.close(fig)
    
    

class weighted_BCE(torch.nn.Module) : 
    def __init__(self, weight=10) : 
        super().__init__()
        self.weight = weight 
    def forward(self, pred, target) : 
        BCE = torch.nn.BCELoss(weight=target*(self.weight-1)+1)
        return BCE(pred, target)
        
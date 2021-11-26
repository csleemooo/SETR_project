import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import PIL
from PIL import Image, ImageFilter
import os
import random
import glob
import pdb
import math
import random
import time
from torchvision.utils import save_image
import torchvision.transforms.functional as tf


def integer_to_channels(target):
    target = (tf.to_tensor(target) * 255).int()
    c, h, w = target.shape
    assert c == 1
    target = target.squeeze(0).numpy()  # (h,w)

    output = np.zeros((h, w, 3), dtype=np.uint8)
    output[:, :, 0] = (target == 1).astype(np.uint8) * 1
    output[:, :, 1] = (target == 2).astype(np.uint8) * 1
    output[:, :, 2] = (target == 3).astype(np.uint8) * 1

    # print(np.unique(output))

    return Image.fromarray(output)

def get_mean_std(images_list):
    pixels = []
    for i, filepath in enumerate(images_list):
        img = Image.open(filepath)
        try:
            img = tf.to_tensor(img)
            pixels.append(img.view(-1))
        except TypeError:
            print(f'{filepath} is truncated')
        if i % 500 == 0:
            print(f'{i}/{len(images_list)}')
        if i == 2000:
            break  # Out of memory..
    pixels = torch.cat(pixels, dim=0)
    return torch.std_mean(pixels, dim=0)

def _random_crop(img1):

    crop_w = 480
    crop_h = 480
    crop_size=(crop_w, crop_h)

    w, h = img1.size
    assert w >= crop_w and h >= crop_h, \
        f'Error: Crop size: {crop_size}, Image size: ({w}, {h})'

    i = np.random.randint(0, h - crop_h + 1)
    j = np.random.randint(0, w - crop_w + 1)

    img1 = tf.crop(img1, i, j, crop_h, crop_w)

    return img1

def sync_transform(*images):

    # random horizontal flip.un
    if random.random() < 0.5 : 

        images = [tf.hflip(image) for image in images]

    # random rotation
    angle = 0
    if random.random() < 0.5:
        angle = random.randint(-15, 15)

    # random scale
    scale = 1
    if random.random() < 0.5:
        scale = random.uniform(480 / 512, 9 / 8)

    images = [tf.affine(image, angle=angle, scale=scale, translate=(0, 0), shear=0,
                        resample=PIL.Image.BILINEAR) for image in images]


    crop_w = 480
    crop_h = 480
    crop_size=(crop_w, crop_h)

    w, h = images[0].size
    assert w >= crop_w and h >= crop_h, \
        f'Error: Crop size: {crop_size}, Image size: ({w}, {h})'

    i = np.random.randint(0, h - crop_h + 1)
    j = np.random.randint(0, w - crop_w + 1)

    images = [tf.crop(img, i, j, crop_h, crop_w) for img in images]    



    return images


class oct_dataset(object):
    def __init__(self, data_path='./oct_data/images', label_path='./oct_data/labels',
                 sync_transform=None, transform=None):
        self.data = []
        self.transform = transform
        self.sync_transform = sync_transform
        self.totensor = torchvision.transforms.ToTensor()

        filenames = os.listdir(data_path)
        self.data = [(os.path.join(data_path, filename), os.path.join(label_path, filename)) for filename in filenames]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path, label_path = self.data[index]
        image_name = os.path.basename(image_path)

        image = Image.open(image_path)
        label = Image.open(label_path)
        label = integer_to_channels(label)

        if self.sync_transform is not None:
            image, label = self.sync_transform(image, label)
        if self.transform is not None:
            image = self.transform(image)
            label = self.transform(label)

        image = image.filter(ImageFilter.MedianFilter(size=5))

        image = self.totensor(image)
        label = self.totensor(label)

        if True:
            image = tf.normalize(image, 0.1410, 0.0941, inplace=True)

        label = label*255

        return image, label, image_name
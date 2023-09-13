from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image, ImageOps
from random import random, randint

# Ignore warnings
import warnings

import pdb

warnings.filterwarnings("ignore")


def make_dataset(root, mode):
    assert mode in ['train','val', 'test','unlabeled']
    items = []

    if mode == 'train':
        train_img_path = os.path.join(root, 'train', 'Img')
        train_mask_path = os.path.join(root, 'train', 'GT')
        
        images = os.listdir(train_img_path)
        labels = os.listdir(train_mask_path)
        
        images.sort()
        labels.sort()
        
        for it_im, it_gt in zip(images, labels):
            item = ('label',os.path.join(train_img_path, it_im), os.path.join(train_mask_path, it_gt))
            items.append(item)
        
    elif mode == 'val':
        val_img_path = os.path.join(root, 'val', 'Img')
        val_mask_path = os.path.join(root, 'val', 'GT')

        images = os.listdir(val_img_path)
        labels = os.listdir(val_mask_path)

        images.sort()
        labels.sort()

        for it_im, it_gt in zip(images, labels):
            item = ("label", os.path.join(val_img_path, it_im), os.path.join(val_mask_path, it_gt))
            items.append(item)
    elif mode=='unlabeled':
        train_unlabeled = os.path.join(root, 'train', 'Img-Unlabeled')
        unlabeled = os.listdir(train_unlabeled)
        unlabeled.sort()
        for i in unlabeled:
            item = ("unlabeled", os.path.join(train_unlabeled, i))
            items.append(item)
        
        #train_img_path = os.path.join(root, 'train', 'Img')
        #train_mask_path = os.path.join(root, 'train', 'GT')
        
        #images = os.listdir(train_img_path)
        #labels = os.listdir(train_mask_path)
        
        #images.sort()
        #labels.sort()
        
        #for it_im, it_gt in zip(images, labels):
            #item = ('label',os.path.join(train_img_path, it_im), os.path.join(train_mask_path, it_gt))
            #items.append(item)
        
    else:
        test_img_path = os.path.join(root, 'test', 'Img')
        test_mask_path = os.path.join(root, 'test', 'GT')

        images = os.listdir(test_img_path)
        labels = os.listdir(test_mask_path)

        images.sort()
        labels.sort()

        for it_im, it_gt in zip(images, labels):
            item = (os.path.join(test_img_path, it_im), os.path.join(test_mask_path, it_gt))
            items.append(item)

    return items


class MedicalImageDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, mode, root_dir, transform=None, mask_transform=None, augment=False, equalize=False):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.imgs = make_dataset(root_dir, mode)
        self.augmentation = augment
        self.equalize = equalize
        self.mode = mode

    def __len__(self):
        return len(self.imgs)

    def augment(self, img, mask):
        if random() > 0.5:
            img = ImageOps.flip(img)
            mask = ImageOps.flip(mask)
        if random() > 0.5:
            img = ImageOps.mirror(img)
            mask = ImageOps.mirror(mask)
        if random() > 0.5:
            angle = random() * 60 - 30
            img = img.rotate(angle)
            mask = mask.rotate(angle)
        return img, mask

    def __getitem__(self, index):
        label=self.imgs[index][0]
        if(label=="label"):
            label,img_path, mask_path = self.imgs[index]
        else:
            label,img_path = self.imgs[index]
        #print(img_path)
        img = Image.open(img_path)
        if(label=="label"):
          mask = Image.open(mask_path).convert('L')
        else :
          mask=0

        if self.equalize:
            img = ImageOps.equalize(img)

        if self.augmentation:
          if(label=="label"):
            img, mask = self.augment(img, mask)
          else:
            img = self.augment(img)

        if self.transform:
            img = self.transform(img)
            if(label=="label"):
              mask = self.mask_transform(mask)

        return [label, img, mask, img_path]
import os
from PIL import Image
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
import cv2
import torch
import random


class test_dataset:
    """load test dataset (batchsize=1)"""

    def __init__(self, pet_root, ct_root, mask_root):


        self.pets = [pet_root + f for f in os.listdir(pet_root) if f.endswith('.png') or f.endswith('jpg') or f.endswith('.bmp')]
        self.cts = [ct_root + f for f in os.listdir(ct_root) if f.endswith('.png') or f.endswith('jpg') or f.endswith('.bmp')]
        self.masks = [mask_root + f for f in os.listdir(mask_root) if f.endswith('.png') or f.endswith('jpg') or f.endswith('.bmp')]

        self.pets = sorted(self.pets)
        self.cts = sorted(self.cts)
        self.masks = sorted(self.masks)

        self.size = len(self.pets)

        self.index = 0


    def load_data(self):
        pet_img = cv2.imread(self.pets[self.index], cv2.IMREAD_GRAYSCALE)  # pet
        ct_img = cv2.imread(self.cts[self.index], cv2.IMREAD_GRAYSCALE)  # ct

        ct_img = np.expand_dims(ct_img, axis=2)
        pet_img = np.expand_dims(pet_img, axis=2)
        ct_img = np.array(ct_img, np.float32).transpose(2, 0, 1) / 255.0 * 3.2 - 1.6
        pet_img = np.array(pet_img, np.float32).transpose(2, 0, 1) / 255.0 * 3.2 - 1.6
        pet_img = pet_img[np.newaxis, :, :, :]
        ct_img = ct_img[np.newaxis, :, :, :]
        pet = torch.tensor(pet_img).repeat(1, 3, 1, 1)
        ct = torch.tensor(ct_img).repeat(1, 3, 1, 1)

        mask = cv2.imread(self.masks[self.index], cv2.IMREAD_GRAYSCALE)

        name = self.masks[self.index].split('/')[-1]

        self.index += 1
        self.index = self.index % self.size

        return pet, ct, mask, name

    def gray_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')


import torch
import random
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import torchvision.transforms as transforms

class Normalize(object):
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['image']
        img = np.array(img).astype(np.float32) / 255.0
        img -= self.mean
        img /= self.std
        sample['image'] = img
        return sample

class ToTensor(object):
    def __call__(self, sample):
        img = sample['image']
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))  # HWC → CHW
        img = torch.from_numpy(img).float()
        if 'gt' in sample:
            mask = np.array(sample['gt']).astype(np.float32) / 255.0
            mask = torch.from_numpy(mask).float()
            return {'image': img, 'gt': mask}
        else:
            return {'image': img}

class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img = sample['image']
        if 'gt' in sample:
            mask = sample['gt']
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if 'gt' in sample:
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        sample['image'] = img
        if 'gt' in sample:
            sample['gt'] = mask
        return sample

class RandomVerticalFlip(object):
    def __call__(self, sample):
        img = sample['image']
        if 'gt' in sample:
            mask = sample['gt']
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            if 'gt' in sample:
                mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
        sample['image'] = img
        if 'gt' in sample:
            sample['gt'] = mask
        return sample

class RandomFixRotate(object):
    def __init__(self):
        self.degree = [Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270]

    def __call__(self, sample):
        img = sample['image']
        if 'gt' in sample:
            mask = sample['gt']
        if random.random() < 0.75:
            rotate_degree = random.choice(self.degree)
            img = img.transpose(rotate_degree)
            if 'gt' in sample:
                mask = mask.transpose(rotate_degree)
        sample['image'] = img
        if 'gt' in sample:
            sample['gt'] = mask
        return sample

class RandomGaussianBlur(object):
    def __call__(self, sample):
        img = sample['image']
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))
        sample['image'] = img
        return sample

class FixedResize(object):
    def __init__(self, size):
        self.size = (size, size)  # (h, w)

    def __call__(self, sample):
        img = sample['image']
        img = img.resize(self.size, Image.BILINEAR)
        sample['image'] = img
        if 'gt' in sample:
            mask = sample['gt']
            assert img.size == mask.size
            mask = mask.resize(self.size, Image.NEAREST)
            sample['gt'] = mask
        return sample

# 训练模式用的数据增强组合
train_transforms = transforms.Compose([
    FixedResize(256),
    RandomHorizontalFlip(),
    RandomVerticalFlip(),
    RandomFixRotate(),
    RandomGaussianBlur(),
    ToTensor()
])

# 推理（测试）模式下的增强（只保留 resize + tensor）
test_transforms = transforms.Compose([
    FixedResize(256),
    ToTensor()
])

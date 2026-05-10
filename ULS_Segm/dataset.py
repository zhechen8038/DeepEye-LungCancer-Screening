import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
from PIL import Image
import random
import pandas as pd
import gc


class CrackData(Dataset):
    def __init__(self, df, transforms=None, mode='train'):
        self.data = df
        self.transform = transforms
        self.mode = mode  # 可为 'train' 或 'test'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.open(self.data['images'].iloc[idx]).convert('RGB')

        if self.mode == 'test':
            # 测试模式：只返回图像和 dummy 标签
            sample = {'image': img}
            if self.transform:
                sample = self.transform(sample)
            return sample['image'], 0  # 你模型 forward 时需要2个值
        else:
            gt = Image.open(self.data['masks'].iloc[idx]).convert('L')
            sample = {'image': img, 'gt': gt}
            if self.transform:
                sample = self.transform(sample)
            return sample['image'], sample['gt']





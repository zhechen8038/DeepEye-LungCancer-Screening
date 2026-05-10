import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import cv2
from tqdm import tqdm
from glob import glob
import pandas as pd
import argparse

from models.seg_model import Seg_Detection
from util.transforms import test_transforms
from dataset import CrackData  # 已支持 mode 参数的版本

# 设置CUDA
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 参数解析
parser = argparse.ArgumentParser('Seg Detection test')
parser.add_argument("--backbone", type=str, default="swinv2_128")
parser.add_argument("--neck", type=str, default="fpn+aspp+fuse+drop")
parser.add_argument("--head", type=str, default="fcn")
parser.add_argument("--pretrain", type=str, default='')
parser.add_argument("--input_size", type=int, default=256)
parser.add_argument("--model_path", type=str, default="./Weights/EMGANet_WHU.pt")
parser.add_argument("--test_path", type=str, default="./dataset/")
parser.add_argument("--save_dir", type=str, default="./save_results/predictions_gray")
opt = parser.parse_args()

# 创建保存目录
os.makedirs(opt.save_dir, exist_ok=True)

# 加载测试图像路径
test_images = sorted(glob(os.path.join(opt.test_path, "img", "*.bmp")))
test_data = pd.DataFrame({'images': test_images})

# 加载测试集（使用 test 模式）
test_dataset = CrackData(df=test_data, transforms=test_transforms, mode='test')
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

# 加载模型（若是保存了完整模型对象）
model = torch.load(opt.model_path, map_location=device)
model = model.to(device)
model.eval()

# 推理并保存灰度图
print("Start inference...")
with torch.no_grad():
    for i, (images, _) in enumerate(tqdm(test_loader)):
        images = images.to(device)
        outputs = model(images)
        if isinstance(outputs, (list, tuple)):
            outputs = outputs[-1]  # 支持多层输出结构，取最后一层

        preds = torch.argmax(outputs, dim=1).squeeze(0).cpu().numpy().astype(np.uint8) * 255
        save_path = os.path.join(opt.save_dir, f"{i:04d}.png")
        cv2.imwrite(save_path, preds)

print("Inference done. Results saved to:", opt.save_dir)

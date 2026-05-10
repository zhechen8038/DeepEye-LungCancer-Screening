import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
from torchvision import transforms
from PIL import Image

from lib.model import CFANet  # 导入模型
print(torch.cuda.is_available())  # 应该输出True
print(torch.version.cuda)  # 查看CUDA版本
# 参数设置
testsize = 352  # 输入分辨率大小
pth_path = './Weights/CFANet.pth'  # 预训练权重路径
image_path = './Datasets/TestDataset/CVC-300/images/150.png'  # 需要预测的图片路径
output_dir = './Save_Single_Map/'  # 预测结果保存目录

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 获取图片文件名（去掉扩展名）
image_name = os.path.basename(image_path)
image_name_no_ext = os.path.splitext(image_name)[0]  # 只获取文件名，不带后缀
save_path = os.path.join(output_dir, f"{image_name_no_ext}.png")  # 结果保存路径

# 创建模型
model = CFANet(channel=64).cuda()
model.load_state_dict(torch.load(pth_path))
model.cuda()
model.eval()

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((testsize, testsize)),  # 调整大小
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 归一化
])

# 读取和预处理图片
with open(image_path, 'rb') as f:
    image = Image.open(f).convert('RGB')
image = transform(image).unsqueeze(0).cuda()  # 增加batch维度

# 前向推理
with torch.no_grad():
    _, _, _, res = model(image)
    res = F.interpolate(res, size=(testsize, testsize), mode='bilinear', align_corners=False)
    res = res.sigmoid().data.cpu().numpy().squeeze()

# 归一化
res = (res - res.min()) / (res.max() - res.min() + 1e-8)

# 保存结果
cv2.imwrite(save_path, res * 255)

print(f"预测完成，结果已保存至: {save_path}")

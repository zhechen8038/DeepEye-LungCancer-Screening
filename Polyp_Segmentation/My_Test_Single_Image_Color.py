import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
from torchvision import transforms
from PIL import Image
from utils.metric import cal_fm, cal_mae, cal_spe, cal_sen, cal_mdice
from lib.model import CFANet  # 导入模型
from utils.heatmap import heatmap

# 参数设置
testsize = 352  # 输入分辨率大小
pth_path = './Weights/CFANet.pth'  # 预训练权重路径
image_path = './Datasets/TestDataset/CVC-300/images/155.png'  # 需要预测的图片路径
gt_path = './Datasets/TestDataset/CVC-300/masks/155.png'
output_dir = './Save_Single_Map/'  # 预测结果保存目录

image_size = cv2.imread(image_path)  # 读取图片
image_size = (image_size.shape[1], image_size.shape[0])  # (宽度, 高度)

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 获取图片文件名（去掉扩展名）
image_name = os.path.basename(image_path)
image_name_no_ext = os.path.splitext(image_name)[0]  # 只获取文件名，不带后缀
save_path = os.path.join(output_dir, f"{image_name_no_ext}.png")  # 预测结果保存路径
save_heat_path = os.path.join(output_dir, f"{image_name_no_ext}_heatmap.png")  # 叠加结果保存路径

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
    res = F.interpolate(res, size=(image_size[1], image_size[0]), mode='bilinear', align_corners=False)
    res = res.sigmoid().data.cpu().numpy().squeeze()

# 归一化
res = (res - res.min()) / (res.max() - res.min() + 1e-8)

# 保存结果
cv2.imwrite(save_path, res * 255)
heatmap(res, image, save_heat_path, image_size)

print(f"预测完成，结果已保存至: {save_path}")
print(f"预测完成，叠加结果已保存至: {save_heat_path}")

# 计算评估指标
def load_image(image_path, grayscale=True):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR)
    return img / 255.0  # 归一化到 0-1 之间

pred = load_image(save_path)
gt = load_image(gt_path)

fm = cal_fm().cal(pred, gt)
mae = cal_mae().cal(pred, gt)
spe = cal_spe().cal(pred, gt)
sen = cal_sen().cal(pred, gt)
mdice = cal_mdice().cal(pred, gt)

print(f"Precision: {fm[0]:.4f}, Recall: {fm[1]:.4f}, F-measure: {fm[2]:.4f}")
print(f"MAE: {mae:.4f}, Specificity: {spe:.4f}, Sensitivity: {sen:.4f}")
print(f"mDice: {mdice:.4f}")

from PIL import Image, ImageDraw, ImageFont

# 读取最终的热力图（使用PIL）
heatmap_img_pil = Image.open(save_heat_path)
draw = ImageDraw.Draw(heatmap_img_pil)

# 设定字体路径（需要确保系统上有该字体文件）
font_path = "C:/Windows/Fonts/simhei.ttf"  # Windows: "C:/Windows/Fonts/simhei.ttf"
font = ImageFont.truetype(font_path, 30)  # 30号字体

# 设定显示位置
text_position_1 = (image_size[0] - 250, 40)  # 右上角
text_position_2 = (image_size[0] - 250, 80)
text_position_3 = (image_size[0] - 250, 120)

draw.text(text_position_1, f"mDice: {mdice:.4f}", font=font, fill=(0, 255, 0))
draw.text(text_position_2, f"SEN: {sen:.4f}", font=font, fill=(0, 255, 0))

# 进行风险提示（中文）
if mdice >= 0.85 and sen >= 0.90:
    risk_message = "高风险：检测到息肉！"
    risk_color = (255, 0, 0)  # 红色
elif sen < 0.75:
    risk_message = "警告：可能存在漏检！"
    risk_color = (255, 165, 0)  # 橙色
else:
    risk_message = "低风险：正常检测。"
    risk_color = (255, 255, 0)  # 黄色

draw.text(text_position_3, risk_message, font=font, fill=risk_color)

# 保存最终带有中文文本的热力图
heatmap_img_pil.save(save_heat_path)
print(f"已更新热力图，风险提示已添加: {save_heat_path}")


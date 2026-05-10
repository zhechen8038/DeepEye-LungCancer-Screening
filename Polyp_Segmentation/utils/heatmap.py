import numpy as np
import cv2
import torch


def heatmap(x_show, img,  path, size):
    # x_show = torch.mean(x_show, dim=1, keepdim=True).data.cpu().numpy().squeeze()
    # x_show = (x_show - x_show.min()) / (x_show.max() - x_show.min() + 1e-8)

    img = img.data.cpu().numpy().squeeze()
    img = img.transpose((1, 2, 0))
    img = img * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    img = img[:, :, ::-1]
    # img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img = np.uint8(255 * img)
    x_show = np.uint8(255 * x_show)
    x_show = cv2.applyColorMap(x_show, cv2.COLORMAP_JET)
    x_show = cv2.resize(x_show, size)
    img = cv2.resize(img, size)
    #print(x_show.shape, img.shape)
    x_show = cv2.addWeighted(img, 0.5, x_show, 0.5, 0)
    #print(x_show.shape)
    #return x_show

    cv2.imwrite(path, x_show)
    # cv2.imshow('img', x_show)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

# import numpy as np
# import cv2
# import torch
#
# def heatmap(x_show, img, path, size):
#     """
#     生成仅前景着色的热力图，并保存到文件
#     :param x_show: 预测结果（单通道，已归一化）
#     :param img: 原始图像（PyTorch 张量，归一化后）
#     :param path: 结果保存路径
#     :param size: 目标尺寸 (width, height)
#     """
#     # 处理原始图片
#     img = img.data.cpu().numpy().squeeze()
#     img = img.transpose((1, 2, 0))  # 从 C×H×W 转为 H×W×C
#     img = img * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))  # 反归一化
#     img = img[:, :, ::-1]  # 转换为 BGR（OpenCV 需要 BGR 格式）
#     img = np.uint8(np.clip(img * 255, 0, 255))  # 限制范围，避免异常值
#
#     # 处理预测结果
#     res_gray = np.uint8(255 * x_show)  # 转换为灰度图
#     heatmap = cv2.applyColorMap(res_gray, cv2.COLORMAP_JET)  # 生成伪彩色图
#     heatmap = cv2.resize(heatmap, size)  # 调整大小
#     img = cv2.resize(img, size)  # 确保原图大小一致
#
#     # 生成二值掩码（基于原始预测）
#     threshold = 0.5  # 设定前景阈值
#     mask = (x_show > threshold).astype(np.uint8) * 255  # 生成二值掩码
#     mask = cv2.resize(mask, size)  # 调整大小
#     mask_3ch = cv2.merge([mask, mask, mask]).astype(np.uint8)  # 确保数据类型匹配
#
#     # 只保留前景的热力图部分
#     foreground = cv2.bitwise_and(heatmap, mask_3ch)  # 仅保留前景的热力图
#     background = cv2.bitwise_and(img, cv2.bitwise_not(mask_3ch))  # 仅保留背景的原图
#
#     # 叠加
#     overlay = cv2.addWeighted(background, 0.5, foreground, 0.5, 0)
#
#     # 保存结果
#     cv2.imwrite(path, overlay)
#     print(f"叠加图已保存至: {path}")


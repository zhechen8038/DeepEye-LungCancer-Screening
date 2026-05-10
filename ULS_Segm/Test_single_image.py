import os
import torch
import numpy as np
import cv2
from PIL import Image
import argparse
from util.transforms import test_transforms  # 你已有的推理transform

def load_model(model_path, device):
    model = torch.load(model_path, map_location=device,weights_only=False)
    model = model.to(device)
    model.eval()
    return model


def preprocess_image(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    sample = {'image': image}
    sample = transform(sample)
    image_tensor = sample['image'].unsqueeze(0)  # batch dim
    return image_tensor, image


def save_results(pred_tensor, orig_image, save_gray_path, save_overlay_path):
    # 预测掩码 (类别索引 * 255)
    pred_mask = torch.argmax(pred_tensor, dim=1).squeeze(0).cpu().numpy().astype(np.uint8) * 255

    # 将灰度掩码resize到原图大小（用NEAREST防止模糊）
    pred_mask_pil = Image.fromarray(pred_mask)
    pred_mask_pil = pred_mask_pil.resize(orig_image.size, Image.NEAREST)
    pred_mask_resized = np.array(pred_mask_pil)

    # 保存灰度图
    cv2.imwrite(save_gray_path, pred_mask_resized)

    # 生成伪彩色热力图
    heatmap = cv2.applyColorMap(pred_mask_resized, cv2.COLORMAP_JET)

    # 原始RGB图转BGR
    orig_rgb = np.array(orig_image)
    orig_bgr = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2BGR)

    # 确保heatmap和原图大小一致
    if heatmap.shape[:2] != orig_bgr.shape[:2]:
        heatmap = cv2.resize(heatmap, (orig_bgr.shape[1], orig_bgr.shape[0]))

    # 叠加
    overlay = cv2.addWeighted(orig_bgr, 0.5, heatmap, 0.5, 0)
    cv2.imwrite(save_overlay_path, overlay)


def infer_single(image_path, model_path, output_dir, input_size=256, device='cuda:0'):
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # 加载模型
    model = load_model(model_path, device)

    # 预处理图像
    transform = test_transforms
    image_tensor, orig_image = preprocess_image(image_path, transform)
    image_tensor = image_tensor.to(device)

    # 推理
    with torch.no_grad():
        output = model(image_tensor)
        if isinstance(output, (list, tuple)):
            output = output[-1]

    # 文件名和路径
    filename = os.path.splitext(os.path.basename(image_path))[0]
    save_gray_path = os.path.join(output_dir, filename + '_gray.png')
    save_overlay_path = os.path.join(output_dir, filename + '_overlay.jpg')

    save_results(output, orig_image, save_gray_path, save_overlay_path)
    print(f"Inference done.\nSaved:\n - {save_gray_path}\n - {save_overlay_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default='./dataset/img/00007.bmp', help='Path to the input image')
    parser.add_argument('--model_path', type=str, default='./Weights/EMGANet_WHU.pt', help='Path to model')
    parser.add_argument('--output_dir', type=str, default='./inference_single_result/',
                        help='Directory to save results')
    parser.add_argument('--input_size', type=int, default=256, help='Resize size')
    parser.add_argument('--device', type=str, default='cuda:0', help='CUDA device or cpu')

    args = parser.parse_args()
    infer_single(args.image_path, args.model_path, args.output_dir, args.input_size, args.device)

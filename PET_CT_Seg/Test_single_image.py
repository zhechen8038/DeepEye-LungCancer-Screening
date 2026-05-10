import os
import cv2
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from Config import parse_args
from models.builder import EncoderDecoder as segmodel
from easydict import EasyDict as edict


def load_model(model_path, device):
    # 固定参数配置
    C = edict()
    C.backbone = 'sigma_tiny'
    C.pretrained_model = None
    C.decoder = 'MambaDecoder'
    C.decoder_embed_dim = 512
    C.image_height = 512
    C.image_width = 512
    C.bn_eps = 1e-3
    C.bn_momentum = 0.1
    C.num_classes = 1

    model = segmodel(cfg=C, norm_layer=torch.nn.BatchNorm2d)
    checkpoint = torch.load(model_path, weights_only=False, map_location=device)
    model.load_state_dict(checkpoint['model'], strict=False)
    model = model.to(device)
    model.eval()
    return model


def preprocess_pet_ct(pet_path, ct_path, device):
    # 读灰度图 + 归一化 + 转tensor + 扩3通道 + batch=1
    pet_img = cv2.imread(pet_path, cv2.IMREAD_GRAYSCALE)
    ct_img = cv2.imread(ct_path, cv2.IMREAD_GRAYSCALE)

    pet_img = np.expand_dims(pet_img, axis=2)
    ct_img = np.expand_dims(ct_img, axis=2)

    pet_img = pet_img.astype(np.float32).transpose(2, 0, 1) / 255.0 * 3.2 - 1.6
    ct_img = ct_img.astype(np.float32).transpose(2, 0, 1) / 255.0 * 3.2 - 1.6

    pet_img = np.expand_dims(pet_img, axis=0)  # batch=1
    ct_img = np.expand_dims(ct_img, axis=0)

    pet = torch.tensor(pet_img).repeat(1, 3, 1, 1).to(device)
    ct = torch.tensor(ct_img).repeat(1, 3, 1, 1).to(device)

    return pet, ct


def save_results(pred, pet_img_path, save_gray_path, save_overlay_path):
    """
    pred: numpy array, shape HxW, 0~255灰度图
    pet_img_path: 用于加载原始PET图，用作叠加底图
    """
    # 读取原PET图，转换为BGR
    pet_img = cv2.imread(pet_img_path)
    if pet_img is None:
        raise RuntimeError(f"PET image {pet_img_path} cannot be read.")

    # 保证尺寸一致，不一致则resize
    if pred.shape != pet_img.shape[:2]:
        pred = cv2.resize(pred, (pet_img.shape[1], pet_img.shape[0]), interpolation=cv2.INTER_NEAREST)

    cv2.imwrite(save_gray_path, pred)

    heatmap = cv2.applyColorMap(pred, cv2.COLORMAP_JET)

    # 叠加
    overlay = cv2.addWeighted(pet_img, 0.5, heatmap, 0.5, 0)
    cv2.imwrite(save_overlay_path, overlay)


def infer_single(pet_path, ct_path, model_path, output_dir, device='cuda:0'):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    model = load_model(model_path, device)

    pet, ct = preprocess_pet_ct(pet_path, ct_path, device)

    with torch.no_grad():
        output = model(ct, pet)  # 注意顺序是model(ct, pet)
        res = torch.sigmoid(output).cpu().numpy()

    pred = np.squeeze(res, axis=0).transpose(1, 2, 0) * 255.0
    pred = pred.astype(np.uint8)

    filename = os.path.splitext(os.path.basename(pet_path))[0]
    save_gray_path = os.path.join(output_dir, filename + '_gray.png')
    save_overlay_path = os.path.join(output_dir, filename + '_overlay.jpg')

    save_results(pred, pet_path, save_gray_path, save_overlay_path)
    print(f"Saved:\n - {save_gray_path}\n - {save_overlay_path}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--pet_path', type=str, default='./data/PET/', help='Path to PET image (grayscale)')
    parser.add_argument('--ct_path', type=str, default='./data/CT/', help='Path to CT image (grayscale)')
    parser.add_argument('--model_path', type=str, default='./weights/CIPA.pth', help='Model weights path')
    parser.add_argument('--output_dir', type=str, default='./save_single_results/', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda:0', help='CUDA device or cpu')

    args = parser.parse_args()

    infer_single(args.pet_path, args.ct_path, args.model_path, args.output_dir, args.device)

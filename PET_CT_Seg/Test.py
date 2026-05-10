from Config import parse_args
from models.builder import EncoderDecoder as segmodel
from PET_CT_Dataset import test_dataset
import cv2
import torch
from easydict import EasyDict as edict
import torch.nn as nn
import os
import numpy as np
import torch.nn.functional as F

cfg = parse_args() #实例化配置类

#模型的一些参数配置
C = edict()
config = C
C.backbone = 'sigma_tiny' # sigma_tiny / sigma_small / sigma_base
C.pretrained_model = None # do not need to change
C.decoder = 'MambaDecoder' # 'MLPDecoder'
C.decoder_embed_dim = 512
C.image_height=512
C.image_width =512
C.bn_eps = 1e-3
C.bn_momentum = 0.1
C.num_classes = 1

model = segmodel(cfg=config, norm_layer=nn.BatchNorm2d)

param = np.sum([p.numel() for p in model.parameters() if p.requires_grad]).item()
print('parameter: %.3fM' % (param/1e6))

checkpoint = torch.load('./weights/CIPA.pth', weights_only=False, map_location=torch.device('cuda'))
model.load_state_dict(checkpoint['model'], strict=False)
model.cuda()
model.eval()

################################################################

save_path = './save_results/'
os.makedirs(save_path, exist_ok=True)

test_loader = test_dataset( pet_root=cfg.test_PET_path, ct_root=cfg.test_CT_path, mask_root=cfg.test_mask_path)

for iteration in range(test_loader.size):
    pet, ct, mask, name = test_loader.load_data()
    #print(name)

    mask = np.asarray(mask, np.float32)
    mask /= (mask.max() + 1e-8)
    #print(mask.shape)

    pet = pet.cuda()
    ct = ct.cuda()

    x_last = model(ct, pet)
    #print(x_last.size())

    #res = F.interpolate(x_last, size=mask.shape[1:], mode='bilinear', align_corners=False)
    res = torch.sigmoid(x_last).detach().cpu().numpy()

    pred = np.squeeze(res, axis=0).transpose(1, 2, 0) * 255.0
    cv2.imwrite(save_path + name, pred)




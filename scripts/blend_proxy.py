import torch
import torch.nn.functional as F
from utils.image_utils import tensor2im
from models.face_parsing.model import BiSeNet, seg_mean, seg_std
from utils.bicubic import BicubicDownSample
from utils.seg_utils import save_vis_mask
from PIL import Image
import numpy as np
from torchvision import transforms as T
import cv2
downsample = BicubicDownSample(factor=2)
img_transform = T.Compose([T.ToTensor(),T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
def hairstyle_feature_blending(generator, seg, latent_bald, bald_F, ref_rotated_w, bald_img):
    w_in = torch.cat([ref_rotated_w[:, :6, :], latent_bald[:, 6:, :]], dim=1)
    hairstyle_feature, _  = generator([w_in], input_is_latent=True, start_layer=0, end_layer=4, randomize_noise=False)
    hairstyle_img, _ = generator([ref_rotated_w], input_is_latent=True, randomize_noise=False)
    hairstyle_img = (downsample((hairstyle_img+1)/2).clamp(0, 1) - seg_mean) / seg_std
    hairstyle_mask = torch.argmax(seg(hairstyle_img)[0], dim=1).unsqueeze(1).long()
    hairstyle_mask = torch.where(hairstyle_mask==10, torch.ones_like(hairstyle_mask), torch.zeros_like(hairstyle_mask))
    hairstyle_mask_down_64 = F.interpolate(hairstyle_mask.float(), size=(64, 64), mode='area')
    
    with torch.no_grad():
        F_blend = hairstyle_feature*hairstyle_mask_down_64+bald_F*(1-hairstyle_mask_down_64)
        img_gen_blend, _= generator(
                [latent_bald],
                input_is_latent=True,
                layer_in=F_blend,
                start_layer=5,
                end_layer=8,
                randomize_noise=False)
        edited_img = tensor2im(img_gen_blend.squeeze())

        hairstyle_mask = (((img_gen_blend+1)/2).clamp(0, 1) - seg_mean) / seg_std
        hairstyle_mask = torch.argmax(seg(hairstyle_mask)[0], dim=1).long().clone().detach()
        ear_mask = torch.where(hairstyle_mask==6, torch.ones_like(hairstyle_mask), torch.zeros_like(hairstyle_mask))[0].cpu().numpy()
        hair_mask = torch.where(hairstyle_mask==10, torch.ones_like(hairstyle_mask), torch.zeros_like(hairstyle_mask))[0].cpu().numpy()
        hair_ear_mask = hair_mask+ear_mask
        face_mask = ((hair_ear_mask.squeeze())*255).astype(np.uint8)
        mask_dilate = cv2.dilate(face_mask,
                                    kernel=np.ones((50, 50), np.uint8))
        mask_dilate_blur = cv2.blur(mask_dilate, ksize=(30, 30))
        mask_dilate_blur = (face_mask + (255 - face_mask) / 255 * mask_dilate_blur).astype(np.uint8)
        
        bald_blending_mask = (255-mask_dilate_blur)
        index = np.where(bald_blending_mask > 0)
        cy = (np.min(index[0]) + np.max(index[0])) // 2
        cx = (np.min(index[1]) + np.max(index[1])) // 2
        center = (cx, cy)
        mixed_clone = cv2.seamlessClone(np.array(bald_img), np.array(edited_img), bald_blending_mask, center, cv2.NORMAL_CLONE)
        edited_img = Image.fromarray(mixed_clone)

    
    return edited_img
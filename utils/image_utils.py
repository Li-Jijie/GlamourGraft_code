import PIL
import torchvision
import torch.nn.functional as F
from PIL import Image
import torch
import torchvision
from PIL import Image
from utils.bicubic import BicubicDownSample
import numpy as np
from models.face_parsing.model import seg_mean, seg_std

import cv2
from torchvision.transforms import transforms
import scipy
import matplotlib.pyplot as plt

def load_image(img_path, normalize=True, downsample=False):
    img = PIL.Image.open(img_path).convert('RGB')
    if downsample:
        img = img.resize((256, 256), PIL.Image.LANCZOS)
    img = transforms.ToTensor()(img)
    if normalize:
        img = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(img)
    return img



def dilate_erosion_mask_path(im_path, seg_net, dilate_erosion=5):
    # # Mask
    # mask = Image.open(mask_path).convert("RGB")
    # mask = mask.resize((256, 256), PIL.Image.NEAREST)
    # mask = transforms.ToTensor()(mask)  # [0, 1]

    IM1 = (BicubicDownSample(factor=2)(torchvision.transforms.ToTensor()(Image.open(im_path))[:3].unsqueeze(0).cuda()).clamp(
        0, 1) - seg_mean) / seg_std
    down_seg1, _, _ = seg_net(IM1)
    mask = torch.argmax(down_seg1, dim=1).long().cpu().float()
    mask = torch.where(mask == 10, torch.ones_like(mask), torch.zeros_like(mask))
    mask = F.interpolate(mask.unsqueeze(0), size=(256, 256), mode='nearest').squeeze()

    # Hair mask + Hair image
    hair_mask = mask
    hair_mask = hair_mask.numpy()
    hair_mask_dilate = scipy.ndimage.binary_dilation(hair_mask, iterations=dilate_erosion)
    hair_mask_erode = scipy.ndimage.binary_erosion(hair_mask, iterations=dilate_erosion)

    hair_mask_dilate = np.expand_dims(hair_mask_dilate, axis=0)
    hair_mask_erode = np.expand_dims(hair_mask_erode, axis=0)

    return torch.from_numpy(hair_mask_dilate).float(), torch.from_numpy(hair_mask_erode).float()

def dilate_erosion_mask_tensor(mask, dilate_erosion=5):
    hair_mask = mask.clone()
    hair_mask = hair_mask.numpy()
    hair_mask_dilate = scipy.ndimage.binary_dilation(hair_mask, iterations=dilate_erosion)
    hair_mask_erode = scipy.ndimage.binary_erosion(hair_mask, iterations=dilate_erosion)

    hair_mask_dilate = np.expand_dims(hair_mask_dilate, axis=0)
    hair_mask_erode = np.expand_dims(hair_mask_erode, axis=0)

    return torch.from_numpy(hair_mask_dilate).float(), torch.from_numpy(hair_mask_erode).float()

def dliate_erode(img, kernel):
    er_k = kernel
    di_k = kernel
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,((er_k//2), (er_k//2)))
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(di_k, di_k))
    img_f = cv2.dilate(img, dilate_kernel)
    img_f = cv2.erode(img_f, erode_kernel)
    return img_f

def tensor2im(var):
    var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
    var = (var + 1) / 2
    var[var < 0] = 0
    var[var > 1] = 1
    var = var * 255
    return Image.fromarray(var.astype("uint8"))

def show_results(src_image, ref_image, bald_image, edited_image):
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    axes[0, 0].imshow(src_image)
    axes[0, 0].set_title('src_image')
    axes[0, 0].axis('off') 

    axes[0, 1].imshow(ref_image)
    axes[0, 1].set_title('ref_image')
    axes[0, 1].axis('off')

    axes[1, 0].imshow(bald_image)
    axes[1, 0].set_title('bald_image')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(edited_image)
    axes[1, 1].set_title('edited_image')
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.show()
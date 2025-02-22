import torch
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms as T
from models.psp.encoders import psp_encoders
from utils.model_utils import get_keys
from models.stylegan2.model import Generator
from utils.image_utils import tensor2im
from models.psp.encoders.psp_encoders import Inverter
import os
import numpy as np
from tqdm import tqdm


class FSEncoder():
    def __init__(self, generator, latent_avg, encoder, device='cuda'):
        self.generator = generator
        self.latent_avg = latent_avg
        self.encoder = encoder
        self.device = device
        self.img_transform = T.Compose([T.ToTensor(),T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    def get_FS(self, x):
        x = self.img_transform(x).unsqueeze(0).to(self.device)
        x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)
        w_recon, predicted_feat = self.encoder.fs_backbone(x)
        w_recon = w_recon + self.latent_avg   
        test_img, _ = self.generator(
            [w_recon],
            input_is_latent=True,
            randomize_noise=False,
        )
        tensor2im(test_img.squeeze()).save('test.jpg')
        w_feat, _ = self.generator(
            [w_recon],
            input_is_latent=True,
            start_layer=0,
            end_layer=4,
            randomize_noise=False,
        )
        fused_feat = self.encoder.fuser(torch.cat([predicted_feat, w_feat], dim=1))
        images, _ = self.generator(
            [w_recon],
            input_is_latent=True,
            layer_in=fused_feat,
            start_layer=5,
            end_layer=8,
            randomize_noise=False
        )
        
        fused_feat = fused_feat.cpu()
        w_feat = w_feat.cpu()
        images = tensor2im(images.squeeze())
        return images, w_recon, fused_feat, w_feat




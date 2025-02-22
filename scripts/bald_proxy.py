import torch
from utils.image_utils import tensor2im
from torchvision import transforms
from models.face_parsing.model import seg_mean, seg_std
import torch.nn.functional as F
from utils.image_utils import dliate_erode
from utils.seg_utils import save_vis_mask
import cv2
import numpy as np
from PIL import Image

class BaldProxy(torch.nn.Module):
    def __init__(self, opts, generator, mapper_for_bald, alpha, seg, editor, fs_encoder):
        super(BaldProxy, self).__init__()
        self.opts = opts
        self.device = self.opts.device
        self.generator = generator
        self.mapper_for_bald = mapper_for_bald
        self.alpha = alpha
        self.img_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        self.seg = seg
        self.editor = editor
        self.encoder = fs_encoder
    def forward(self, src_img, src_e4e_w, src_sfe_w, src_sfe_F, possion=False):
        w_bald_mapper = src_e4e_w.clone().detach()
        with torch.no_grad():
            w_bald_mapper[:, :8, :] += self.alpha * self.mapper_for_bald(w_bald_mapper)
            w_bald_mapper[:, 7:18,:] = 0.8*src_e4e_w[:, 7:18,:]+0.2*w_bald_mapper[:, 7:18,:]
            mapper_bald_feature_64, _ = self.generator([w_bald_mapper], input_is_latent=True,  randomize_noise = False, start_layer=0, end_layer=4)
            mapper_bald_feature_32, _ = self.generator([w_bald_mapper], input_is_latent=True,  randomize_noise = False, start_layer=0, end_layer=3)
            src_tensor = self.img_transform(src_img).to(self.device)
            src_tensor_seg = (((src_tensor+1)/2).clamp(0, 1) - seg_mean) / seg_std
            src_mask = torch.argmax(self.seg(src_tensor_seg)[0], dim=1).long().clone().detach()
            ear_mask = torch.where(src_mask==6, torch.ones_like(src_mask), torch.zeros_like(src_mask))[0].cpu().numpy()
            hair_mask = torch.where(src_mask==10, torch.ones_like(src_mask), torch.zeros_like(src_mask))[0].cpu().numpy()
            hair_ear_mask = ear_mask + hair_mask
            bald_blending_mask = dliate_erode(hair_ear_mask.astype('uint8'), 30)
            bald_blending_mask = torch.from_numpy(bald_blending_mask).unsqueeze(0).unsqueeze(0).cuda()
            bald_blending_mask_down = F.interpolate(bald_blending_mask.float(), size=(64, 64), mode='area')
            delta = src_sfe_F*bald_blending_mask_down - mapper_bald_feature_64*bald_blending_mask_down
            F_bald_blend = torch.cat([src_sfe_F, delta], dim = 1)
            F_bald_edited = self.editor(F_bald_blend)
            bald_output, _ = self.generator(
                        [src_sfe_w],
                        input_is_latent=True,
                        start_layer=5,
                        end_layer=8,
                        layer_in=F_bald_edited,
                        randomize_noise=False)
            bald_img = tensor2im(bald_output.squeeze())
            w_bald = w_bald_mapper.detach().clone()
            if possion:
                face_mask = ((hair_ear_mask)*255).astype(np.uint8)
                mask_dilate = cv2.dilate(face_mask,
                                            kernel=np.ones((50, 50), np.uint8))
                mask_dilate_blur = cv2.blur(mask_dilate, ksize=(30, 30))
                mask_dilate_blur = (face_mask + (255 - face_mask) / 255 * mask_dilate_blur).astype(np.uint8)
                
                bald_blending_mask = (255-mask_dilate_blur)
                index = np.where(bald_blending_mask > 0)
                cy = (np.min(index[0]) + np.max(index[0])) // 2
                cx = (np.min(index[1]) + np.max(index[1])) // 2
                center = (cx, cy)
                mixed_clone = cv2.seamlessClone(np.array(src_img), np.array(bald_img), bald_blending_mask, center, cv2.NORMAL_CLONE)
                possion_bald_img = Image.fromarray(mixed_clone)
                return src_sfe_w, w_bald, mapper_bald_feature_32, F_bald_edited, possion_bald_img
            return src_sfe_w, w_bald, mapper_bald_feature_32, F_bald_edited, bald_img
                
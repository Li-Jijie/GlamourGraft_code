import numpy as np
import torch
import torch.nn.functional as F
from models.face_parsing.model import seg_mean, seg_std
from tqdm import tqdm
from criteria.transfer_loss import TransferLossBuilder
from torchvision import transforms as T
from utils.bicubic import BicubicDownSample
from utils.image_utils import tensor2im
from criteria.id_loss import IDLoss

class AlignProxy(torch.nn.Module):
    def __init__(self, opts, generator, seg, kp_extractor, latent_avg, align_mapper):
        super(AlignProxy, self).__init__()
        self.opts = opts
        self.generator = generator
        self.mean_latent = latent_avg
        self.seg = seg
        self.kp_extractor = kp_extractor
        self.align_mapper = align_mapper
        self.transfer_loss_builder = TransferLossBuilder()
        self.delta_loss = torch.nn.MSELoss()
        self.landmark_loss = torch.nn.MSELoss()
        self.mask_loss = self.weighted_ce_loss()
        self.create_loss = torch.nn.CrossEntropyLoss()
        self.downsample_256 = BicubicDownSample(factor=4)
        self.downsample = BicubicDownSample(factor=2)
        self.toLandmarks = T.Compose([
                    T.Resize((256, 256)),
                    T.Normalize(0.5, 0.5)
                ])
        self.to_tensor = T.ToTensor()
        self.img_transform = T.Compose([T.ToTensor(),T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        self.loss_id = IDLoss()
    
    def weighted_ce_loss(self):
        weight_tmp = torch.zeros(16).cuda()
        weight_tmp[10] = 1
        weight_tmp[1] = 1
        weight_tmp[6] = 1
        weight_tmp[0] = 1
        return torch.nn.CrossEntropyLoss(weight=weight_tmp).cuda()

    
    def preprocess_img(self, img_tensor):
        img_tensor = (img_tensor+1)/2
        im = (self.downsample(img_tensor).clamp(0, 1) - seg_mean) / seg_std
        return im
    
    def inference_on_kp_extractor(self, input_image):
        return self.kp_extractor.face_alignment_net(((F.interpolate(input_image, size=(256, 256)) + 1) / 2).clamp(0, 1))

    def gen_256_img_hairmask(self, input_image): 
        input_image = (((input_image+1)/2).clamp(0, 1) - seg_mean) / seg_std
        input_seg = torch.argmax(self.seg(input_image)[0].clone().detach(), dim=1).long()
        input_hairmask = torch.where((input_seg == 10), torch.ones_like(input_seg), torch.zeros_like(input_seg))
        input_hairmask_256 = F.interpolate(input_hairmask.unsqueeze(0).float(), size=(256, 256), mode='area')
        input_img_256 = F.interpolate(input_image, size=(256, 256))
        return input_img_256, input_hairmask_256

    def setup_align_optimizer(self, latent_path=None):
        if latent_path:
            latent_W = torch.from_numpy(np.load(latent_path)['latent_in']).cuda().requires_grad_(True)
        else:
            latent_W = self.mean_latent.reshape(1, 1, 512).repeat(1, 18, 1).clone().detach().to(self.opts.device).requires_grad_(True)

        optimizer_align = torch.optim.Adam([latent_W], lr=self.opts.lr_ref)

        return optimizer_align, latent_W
    
    def create_down_seg(self, gen_im):
        gen_im_0_1 = (gen_im + 1) / 2
        im = (self.downsample(gen_im_0_1).clamp(0, 1) - seg_mean) / seg_std
        down_seg, _, _ = self.seg(im)
        return down_seg, gen_im
    
    def create_target_segmentation_mask(self, src_tensor, ref_tensor, src_name, ref_name):

        down_seg1 = self.create_down_seg(src_tensor)[0]
        seg_target1 = torch.argmax(down_seg1, dim=1).long()
        down_seg2 = self.create_down_seg(ref_tensor)[0]
        seg_target2 = torch.argmax(down_seg2, dim=1).long()    
        target_mask = torch.where(seg_target2==10, seg_target2, seg_target1)
        target_mask = torch.where((seg_target2==0)*(seg_target1==1), 10*torch.ones_like(seg_target1), target_mask)

        optimizer_align, latent_align = self.setup_align_optimizer()
        latent_end = latent_align[:, 6:, :].clone().detach()

        pbar = tqdm(range(self.opts.mask_align_step), desc='Create Target Mask Step')
        
        for step in pbar:
            optimizer_align.zero_grad()
            latent_in = torch.cat([latent_align[:, :6, :], latent_end], dim=1)
            gen_im, _ = self.generator([latent_in], input_is_latent=True, return_latents=False, randomize_noise=False)
            down_seg, _ = self.create_down_seg(gen_im)
            ce_loss = self.create_loss(down_seg, target_mask)
            loss = ce_loss
            loss.backward()
            optimizer_align.step()

        gen_seg_target = torch.argmax(down_seg, dim=1).long()
        target_mask = torch.where(gen_seg_target==10, gen_seg_target, seg_target1)
        target_mask = torch.where((gen_seg_target==0)*(seg_target1==1), 10*torch.ones_like(gen_seg_target), target_mask)
        target_mask = torch.where((target_mask==0)*((gen_seg_target==1)|(gen_seg_target==6)|(gen_seg_target==10)), 10*torch.ones_like(gen_seg_target), target_mask)
        
        return target_mask.long().cuda()

    
    def generate_key_points(self, batch):
        batch = (batch + 1)/2
        batch = self.downsample_256(batch).clip(0, 1)
        toLandmarks = T.Compose([T.Resize((256, 256)),T.Normalize(0.5, 0.5)])
        _, _, landmarks = self.kp_extractor(toLandmarks(batch))
        final_marks_2D = (landmarks[:, :] + 1) / 2 * torch.tensor([256 - 1, 256 - 1]).to('cuda').view(1, 1, 2)
        return final_marks_2D

    def forward(self, hairstyle_image, bald_image, bald_w, ref_w, src_name, ref_name):
        bald_w = bald_w.squeeze(1)
        ref_w = ref_w.squeeze(1)

        w_rotate = self.align_mapper(ref_w[:, :6, :], bald_w[:, :6, :])
        ref_w = torch.cat([w_rotate, ref_w[:, 6:, :]], dim = 1)

        ref_tensor = self.img_transform(hairstyle_image).unsqueeze(0).to(self.opts.device)
        latent_W_optimized = ref_w.clone().detach().cuda().requires_grad_(True)
        ref_w_pre = latent_W_optimized[:, :6, :].clone().detach()
        ref_w_back = latent_W_optimized[:, 6:, :].clone().detach()
        ref_img_256, ref_hairmask_256 = self.gen_256_img_hairmask(ref_tensor)
        optimizer = torch.optim.Adam([latent_W_optimized], lr=self.opts.lr_ref)
        ori_tensor = self.to_tensor(bald_image).unsqueeze(0).to(self.opts.device)
        src_kp = self.generate_key_points(ori_tensor).clone().detach()
        pbar = tqdm(range(self.opts.steps_ref))
        for i in pbar:
            optimizer.zero_grad()
            latent_in = torch.cat([latent_W_optimized[:, :6, :], ref_w_back], dim=1)
            img_gen, _ = self.generator([latent_in], input_is_latent=True, return_latents=False, randomize_noise=False)
            gen_kp = self.generate_key_points(img_gen)
            down_seg, _ = self.create_down_seg(img_gen)
            img_gen_256, gen_hairmask_256 = self.gen_256_img_hairmask(img_gen)
            if(i == self.opts.steps_mask):
                target_mask= self.create_target_segmentation_mask(ori_tensor, img_gen, src_name, ref_name).clone().detach()
            hair_style_loss = self.transfer_loss_builder.style_loss(ref_img_256, img_gen_256, mask1=ref_hairmask_256, mask2=gen_hairmask_256)
            delta_w_loss = self.delta_loss(latent_W_optimized[:, :6, :], ref_w_pre)
            kp_loss = self.landmark_loss(gen_kp, src_kp)
            if(i >= self.opts.steps_mask):
                loss_mask = self.mask_loss(down_seg, target_mask)
                loss = self.opts.lambda_mask*loss_mask + self.opts.lambda_style*hair_style_loss + self.opts.lambda_delta*delta_w_loss 
            else:
                loss = self.opts.lambda_kp*kp_loss + self.opts.lambda_style*hair_style_loss + self.opts.lambda_delta*delta_w_loss
            loss.backward()
            optimizer.step()
            ref_w_pre = latent_W_optimized[:, :6, :].clone().detach()
            pbar.set_description((f"ref_loss: {loss.item():.4f};"))
            
            if i == (self.opts.steps_ref-1):
                with torch.no_grad():
                    latent_in = torch.cat([latent_W_optimized[:, :6, :], ref_w_back], dim=1)
                    gen_aligned, _ = self.generator([latent_in], input_is_latent=True, return_latents=False, randomize_noise=False)
                    gen_aligned_vis = tensor2im(gen_aligned.squeeze())

        return gen_aligned_vis, latent_in.clone().detach()
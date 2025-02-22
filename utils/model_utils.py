#from models.psp.stylegan2.model import Generator
from models.stylegan2.model import Generator
from models.face_parsing.model import BiSeNet
from models.bald_proxy.networks.level_mapper import LevelMapper
import torch
from models.psp.encoders.psp_encoders import Inverter
from models.STAR.lib import utility
import argparse
from models.align_net import RotateModel

def get_keys(d, name):
	if 'state_dict' in d:
		d = d['state_dict']
	d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
	return d_filt

def toggle_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def load_models(opts):

    #Load stylegan2 model
    generator = Generator(1024, 512, 8).to(opts.device)
    ckpt_gan = torch.load(opts.stylegan_path)
    generator.load_state_dict(ckpt_gan["g_ema"], strict=True)
    latent_avg = ckpt_gan["latent_avg"].to(opts.device)
    generator.eval()

    #Load segmentation model
    seg_net = BiSeNet(n_classes=16).to(opts.device)
    seg_net.load_state_dict(torch.load(opts.seg_path))
    for param in seg_net.parameters():
        param.requires_grad = False
    seg_net.eval()

    #Load face alignment model
    args = argparse.Namespace()
    args.config_name = 'alignment'
    config = utility.get_config(args)
    kp_extractor = utility.get_net(config)
    checkpoint = torch.load(opts.kp_path)
    kp_extractor.load_state_dict(checkpoint["net"])
    kp_extractor = kp_extractor.float().to(opts.device)
    kp_extractor.eval()
    
    #Load balding model
    bald_mapper = LevelMapper(input_dim=512).to(opts.device)
    bald_ckpt = torch.load(opts.bald_path)
    alpha = float(bald_ckpt['alpha']) * 1.2
    bald_mapper.load_state_dict(bald_ckpt['state_dict'], strict=True)
    bald_mapper.eval()

    sfe_encoder = Inverter(n_styles=18).to(opts.device)
    sfe_ckpt = torch.load(opts.sfe_encoder_path)
    sfe_encoder.load_state_dict(get_keys(sfe_ckpt, "encoder"), strict=True)
    sfe_encoder.eval()

    bald_editor = torch.load(opts.bald_editor_path)
    bald_editor.eval()


    align_mapper = RotateModel().to(opts.device)
    align_mapper.load_state_dict(torch.load('pretrained_models/rotate_best.pth')['model_state_dict'], strict=False)
    align_mapper.eval()

    toggle_grad(generator, False)
    toggle_grad(seg_net, False)
    toggle_grad(kp_extractor, False)
    toggle_grad(bald_mapper, False)
    toggle_grad(bald_editor, False)
    toggle_grad(sfe_encoder, False)  
    toggle_grad(align_mapper, False)     
    return generator, latent_avg, seg_net, kp_extractor, bald_mapper, alpha, bald_editor, sfe_encoder, align_mapper

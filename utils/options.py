from argparse import ArgumentParser


class Options:

    def __init__(self):
        self.parser = ArgumentParser()
        self.initialize()

    def initialize(self):
        
        # arguments for device
        self.parser.add_argument('--device', default="cuda", type=str, help='Device for training and inference')
        # arguments for pretrained model path
        self.parser.add_argument('--stylegan_path', default="pretrained_models/stylegan2-ffhq-config-f.pt", type=str, help='Path to StyleGAN model checkpoint')
        self.parser.add_argument('--seg_path', default="pretrained_models/seg.pth", type=str, help='Path to face parsing model checkpoint')
        self.parser.add_argument('--bald_path', default="pretrained_models/bald_proxy.pt", type=str, help='Path to balding model checkpoint')
        self.parser.add_argument('--bald_editor_path', default="pretrained_models/bald_editor.pt", type=str, help='Path to bald editor checkpoint')
        self.parser.add_argument('--e4e_path', default='pretrained_models/e4e_ckpt.pt', type=str, help='Path to E4E Encoder checkpoint')
        self.parser.add_argument('--kp_path', default='pretrained_models/WFLW_STARLoss_NME_4_02_FR_2_32_AUC_0_605.pkl', type=str, help='Path to Keypoint Extractor checkpoint')
        # arguments for image and latent dir path
        self.parser.add_argument('--src_img_dir', default="images/src_images/", type=str, help='Folder of source image')     
        self.parser.add_argument('--ref_img_dir', default="images/ref_images/", type=str, help='Folder of reference image')
        self.parser.add_argument('--edited_img_dir', default="images/edited_images/", type=str, help='Folder of edited image')
        # arguments for align 
        self.parser.add_argument('--lr_ref', default=0.01, type=float, help='Learning rate for Align proxy')
        self.parser.add_argument('--mask_align_step', default=80, type=int, help='Step for Mask Align optimization')
        self.parser.add_argument('--steps_ref', default=100, type=int, help='Step for Align proxy optimization')
        self.parser.add_argument('--steps_mask', default=40, type=int, help='Step for beginning FineMask alignment')
        self.parser.add_argument('--arcface_model_path', default='pretrained_models/iresnet50-7f187506.pth', type=str, help='Step for Align proxy optimization')
        self.parser.add_argument('--sfe_encoder_path', default='pretrained_models/sfe_inverter_light.pt', type=str, help='Path to sfe inverter checkpoint')
        self.parser.add_argument('--lambda_style', default=4e4, type=int, help='Weight for style loss')
        self.parser.add_argument('--lambda_delta', default=1000, type=int, help='Weight for delta loss')
        self.parser.add_argument('--lambda_kp', default=1000, type=int, help='Weight for keypoint loss')
        self.parser.add_argument('--lambda_mask', default=1e5, type=int, help='Weight for mask loss')
    def parse(self, jupyter=False):
        if jupyter:
            opts = self.parser.parse_args(args=[])
        else:
            opts = self.parser.parse_args()
        return opts


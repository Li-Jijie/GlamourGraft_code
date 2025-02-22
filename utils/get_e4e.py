import sys
sys.path.append('./')
from models.psp.encoders.psp_encoders import Encoder4Editing
import torch
from utils.model_utils import get_keys
from torchvision import transforms as T
class E4EEncoder():
    def __init__(self, e4e_ckpt, latent_avg, device='cuda'):
        self.device = device
        self.latent_avg = latent_avg
        ckpt = torch.load(e4e_ckpt)
        self.e4e_encoder = Encoder4Editing(50, 'ir_se')
        self.e4e_encoder.load_state_dict(get_keys(ckpt, 'encoder'), strict=True)
        self.e4e_encoder.eval().to(self.device)
        self.img_transform = T.Compose([
                        T.Resize((256, 256)),
                        T.ToTensor(),
                        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    def get_e4e(self, img):

        img_tensor = self.img_transform(img).to(self.device).unsqueeze(0)
        latent_code = self.e4e_encoder(img_tensor)
        latent_code = latent_code + self.latent_avg.repeat(latent_code.shape[0], 1, 1)

        return latent_code


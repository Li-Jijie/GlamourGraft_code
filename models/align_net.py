from models.psp.stylegan2.model import PixelNorm, EqualLinear
import torch
from torch import nn
from torch.nn import Module
from torch.nn import Linear, LayerNorm, LeakyReLU, Sequential
from PIL import Image
import torchvision.transforms as T

class ModulationModule(nn.Module):
    def __init__(self, layernum, last=False, inp=512, middle=512):
        super().__init__()
        self.layernum = layernum
        self.last = last
        self.fc = Linear(512, 512)
        self.norm = LayerNorm([self.layernum, 512], elementwise_affine=False)
        self.gamma_function = Sequential(Linear(inp, middle), LayerNorm([middle]), LeakyReLU(), Linear(middle, 512))
        self.beta_function = Sequential(Linear(inp, middle), LayerNorm([middle]), LeakyReLU(), Linear(middle, 512))
        self.leakyrelu = LeakyReLU()

    def forward(self, x, embedding):
        x = self.fc(x)
        x = self.norm(x)
        gamma = self.gamma_function(embedding)
        beta = self.beta_function(embedding)
        out = x * (1 + gamma) + beta
        if not self.last:
            out = self.leakyrelu(out)
        return out


class RotateModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.pixelnorm = PixelNorm()
        self.modulation_module_list = nn.ModuleList([ModulationModule(6, i == 4) for i in range(5)])

    def forward(self, latent_from, latent_to):
        dt_latent = self.pixelnorm(latent_from)
        for modulation_module in self.modulation_module_list:
            dt_latent = modulation_module(dt_latent, latent_to)
        output = latent_from + 0.1 * dt_latent
        return output
    

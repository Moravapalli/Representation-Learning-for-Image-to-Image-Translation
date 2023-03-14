#module for reconstruction

import torch
import torch.nn as nn

from modules.architecture import Decoder
from modules.vector_qan import VectorQuantizer2 as VectorQuantizer

class Re_con(nn.Module):
    def __init__(self, args):
        super(Re_con, self).__init__()
        self.decoder = Decoder(args)
        self.quantize = VectorQuantizer(args)
        self.post_quant_conv = nn.Conv2d(args.latent_dim,args.latent_dim,1)
        self.quant_conv = nn.Conv2d(args.latent_dim, args.latent_dim, 1)

    def forward(self,lat):
        lat = self.quant_conv(lat)
        quant,code_loss, info = self.quantize(lat)
        lat = self.post_quant_conv(quant)
        dec_img = self.decoder(lat)
        return dec_img


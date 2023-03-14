import torch
import torch.nn as nn
from modules.architecture import Encoder,Decoder
from modules.vector_qan import VectorQuantizer2 as VectorQuantizer

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.encoder = Encoder(args)
        #self.quantize = VectorQuantizer(args)
        #self.quant_conv = nn.Conv2d(args.latent_dim,args.latent_dim,1)

    def forward(self,image):
        latent = self.encoder(image)
        #quant = self.quant_conv(i)
        #code_vec,code_loss, info = self.quantize(quant)
        return latent


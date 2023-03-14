from PIL import Image
import os
import torch
from torch.utils.data import Dataset
import numpy as np
#import albumentations as A
#from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy
from torchvision.utils import save_image

class CustomDataset(Dataset):
    def __init__(self, root_syn, root_real, transform=None):
        #self.root_zebra = root_zebra
        #self.root_horse = root_horse
        self.syn = [file.strip() for file in open(root_syn, 'r')]
        self.real = [y.strip() for y in open(root_real, 'r')]
        self.transform = transform
        
        #self.zebra_images = os.listdir(root_zebra)
        #self.horse_images = os.listdir(root_horse)
        self.length_dataset = max(len(self.syn), len(self.real))
        self.zebra_len = len(self.syn)
        self.horse_len = len(self.real)
        #self.length_dataset = max(len(self.zebra_images), len(self.horse_images)) # 1000, 1500
        #self.zebra_len = len(self.zebra_images)
        #self.horse_len = len(self.horse_images)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        syn_img = self.syn[index]# % self.zebra_len]
        real_img = self.real[index]# % self.horse_len]

        #zebra_path = os.path.join(self.root_zebra, zebra_img)
        #horse_path = os.path.join(self.root_horse, horse_img)

        syn_img = np.array(Image.open(syn_img).convert("RGB"))
        real_img = np.array(Image.open(real_img).convert("RGB"))
        syn_img = (syn_img - np.min(syn_img)) / (np.max(syn_img) - np.min(syn_img))
        real_img = (real_img - np.min(real_img)) / (np.max(real_img) - np.min(real_img))
        if self.transform:
            augmentations = self.transform(image=syn_img, image0=real_img)
            syn_img = augmentations["image"]
            real_img = augmentations["image0"]

        return syn_img, real_img


'''def gradient_penalty( critic, real_image, fake_image, device="cpu"):
    batch_size, channel, height, width= real_image.shape
    #alpha is selected randomly between 0 and 1
    alpha= torch.rand(batch_size,1,1,1).repeat(1, channel, height, width)
    # interpolated image=randomly weighted average between a real and fake image
    #interpolated image ← alpha *real image  + (1 − alpha) fake image
    interpolatted_image=(alpha*real_image) + (1-alpha) * fake_image
    
    # calculate the critic score on the interpolated image
    interpolated_score= critic(interpolatted_image)
    
    # take the gradient of the score wrt to the interpolated image
    gradient= torch.autograd.grad(inputs=interpolatted_image,
                                  outputs=interpolated_score,
                                  retain_graph=True,
                                  create_graph=True,
                                  grad_outputs=torch.ones_like(interpolated_score)                          
                                 )[0]
    gradient= gradient.view(gradient.shape[0],-1)
    gradient_norm= gradient.norm(2,dim=1)
    gradient_penalty=torch.mean((gradient_norm-1)**2)
    return gradient_penalty'''

def calculate_gradient_penalty(model, real_images, fake_images, device):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake data
    alpha = torch.randn((real_images.size(0), 1, 1, 1), device=device)
    # Get random interpolation between real and fake data
    interpolates = (alpha * real_images + ((1 - alpha) * fake_images)).requires_grad_(True)

    model_interpolates = model(interpolates)
    grad_outputs = torch.ones(model_interpolates.size(), device=device, requires_grad=False)

    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=model_interpolates,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = torch.mean((gradients.norm(2, dim=1) - 1) ** 2)
    return gradient_penalty
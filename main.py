import torch
import sys
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision.utils import save_image

from modules.Discriminator import Discriminator
from modules.encoder_model import Model
from modules.architecture import  Encoder
from modules.recon import Re_con
from dataset import CustomDataset, calculate_gradient_penalty
import config

import wandb
wandb.login(key='6b037e771b020c9d419f6468780b6a0640a9e9eb')

def train(recon,r_con,disc, syn_enc, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler,epoch):

    total_samples = len(loader)
    LAMBDA_GP = 10
    loop = tqdm(loader, leave=True)
    for idx, (syn, real) in enumerate(loop):

        # get the input data(synthetic and real images)
        syn = syn.to(config.DEVICE, dtype=torch.float)
        real = real.to(config.DEVICE, dtype=torch.float)

        # forward
        real_lat = recon(real)

        # Train Discriminator
        disc.zero_grad()
        # train with real
        real_disc = disc(real_lat)
        err_real = torch.mean(real_disc)
        D_x = real_disc.mean().item()
        
        # train with fake 
        fake_lat = syn_enc(syn)
        fake_disc = disc(fake_lat.detach())
        err_fake = torch.mean(fake_disc)
        D_G1 = fake_disc.mean().item()
        
        #cal gradient penality(gp)
        gp = calculate_gradient_penalty(disc,real_lat,fake_lat,config.DEVICE)

        #Add the gradients from all the real and fake batches
        errD = -err_real+err_fake+gp*10
        errD.backward()
        opt_disc.step()

        # train the generator every n critic iterations

        if (idx + 1) % config.critic_iter == 0:
            syn_enc.zero_grad()

            # Generate fake latent
            fake_lat = syn_enc(syn)
            fake_disc = disc(fake_lat)

            errG = -torch.mean(fake_disc)
            D_G2 = fake_lat.mean().item()
            errG.backward()
            opt_gen.step()
            loop.set_postfix(Epoch= epoch,D_loss=errD, G_loss=errG,D_x = D_x,D_G1 = D_G1,D_G2 = D_G2)

            # log training data to the wandb
            wandb.log({'D_loss':errD, 'G_loss':errG,'D_x' :D_x,'D_G1' : D_G1,'D_G2' : D_G2})

        # saving weights and reconstruted images
        if epoch % 100 == 0 and idx == total_samples//config.BATCH_SIZE-5:
            img = r_con(fake_lat)
            fake_z = torch.cat((syn[:2], img[:2]))
            save_image(fake_z, f"/home/moravapa/Documents/cycle/saved_images/1/GP/gp_{epoch}.png")
            torch.save(syn_enc.state_dict(), f"/home/moravapa/Documents/cycle/saved_images/1/GP/gp_{epoch}.pt")

        
def Trainer():
    disc = Discriminator().to(config.DEVICE)
    syn_enc = Encoder(config).to(config.DEVICE)        
    real_encoder = Model(config) 
    r_con = Re_con(config)

    # Load real-encoder with the weights trained on real images 
    if config.load_real:
        checkpoint2 = torch.load(config.checkpoint2, map_location='cuda')
        real_list = ['encoder']
        pretrained_dict = {x: y for x, y in checkpoint2['model_state_dict'].items() if x.split('.', 1)[0] in real_list}
        real_encoder.load_state_dict(pretrained_dict)
        real_encoder.eval()
        real_encoder.requires_grad_(False)
        real_encoder = real_encoder.to(config.DEVICE)
        print('real_encoder', real_encoder.training)

    # Loading weights to check the reconstructions after mentioned epochs    
    if config.load_recon:
        recon_list = ['decoder', 'post_quant_conv','quant_conv', 'quantize']
        pretrained_dict = {x: y for x, y in checkpoint2['model_state_dict'].items() if x.split('.', 1)[0] in recon_list}
        r_con.load_state_dict(pretrained_dict)
        r_con.eval()
        r_con.requires_grad_(False)
        r_con = r_con.to(config.DEVICE)
        print('r_con', r_con.training)



    #  Define the optimizer for discriminator and synthetic_encoder
    opt_disc = optim.Adam(disc.parameters(),
                          lr=config.LEARNING_RATE,
                          betas=(0.5, 0.999))
    opt_gen = optim.Adam(syn_enc.parameters(),
                         lr=config.LEARNING_RATE,
                         betas=(0.5, 0.999))
    
    # Define the loss function(not utilized to this)
    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    # Load the cityscape and GTA datasets
    dataset = CustomDataset(
        root_real=r'/home/moravapa/Documents/Taming/data/train_1.txt', root_syn=r'/home/moravapa/Documents/Taming/data/train_syn.txt', transform=config.transforms
    )
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    # Initialize the GradScalers 
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    # Train the model
    for epoch in range(config.NUM_EPOCHS):
        train(real_encoder,r_con,disc, syn_enc, loader, opt_disc, opt_gen, L1, mse, d_scaler, g_scaler,epoch)


if __name__ == "__main__":
    wandb.init(project="Experiments", name="project", dir='Experiments')
    wandb.run.name = 'Using_GP'
    Trainer()
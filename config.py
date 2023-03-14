import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
BATCH_SIZE = 6
LEARNING_RATE = 1e-4
LAMBDA_IDENTITY = 0.5
LAMBDA_CYCLE = 10
NUM_WORKERS = 4
NUM_EPOCHS = 801
in_channels = 128
out_channels = 3
ch_mult = [1,1,2,2,4]
resolution = 256
image_channels = 3
num_codebook_vectors = 1024
latent_dim = 256
image_size = 256
dropout = 0.0
load_syn = False
load_real = True
load_recon =True
beta = 0.25
critic_iter = 5
checkpoint1 = r'/home/moravapa/Documents/ckpt/vqsyn_200.pt'
checkpoint2 = r"/home/moravapa/Documents/ckpt/vqreal_200.pt"
LOAD_MODEL = False
SAVE_MODEL = False
CHECKPOINT_gen_real = "/home/moravapa/Documents/cycle/saved_images/1/cycle_latent/img2lat/cycle_img2lat.pth"
CHECKPOINT_gen_fake = "/home/moravapa/Documents/cycle/saved_images/1/cycle_latent/img2lat/cycle_lat2img.pth"
CHECKPOINT_CRITIC_H = "saved_images/ckpt/cycle_disc.pth"
CHECKPOINT_CRITIC_Z = "saved_images/ckpt/cycle_disc.pth"
transforms = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        ToTensorV2(),
     ],
    additional_targets={"image0": "image"},
)
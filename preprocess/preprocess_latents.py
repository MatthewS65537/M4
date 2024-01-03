import sys
sys.path.append("./models")
sys.path.append("./training")
sys.path.append("./testing")
sys.path.append("./utils")
sys.path.append("./ZuCo")
sys.path.append("./trainer")
import time

# Import Pytorch
import torch
import torch.nn as nn

# Import Pretrain Libraries (transformers + diffusers)
from transformers import BartTokenizer
from diffusers import AutoencoderKL

dataset_dict = INITIALIZE_DATALOADERS(
    keys=["Brain2Image"],
    bsz=[1],
    dev_bsz=[1]
)

vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").to("cuda", dtype=torch.float32)
vae = vae.requires_grad_(False)

latent_dict = {}

from PIL import Image
from torchvision import transforms as tfms

to_tensor_tfm = tfms.ToTensor()

def pil_to_latent(vae, input_im):
  # Single image -> single latent in a batch (so size 1, 4, 64, 64)
  with torch.no_grad():
    latent = vae.encode(to_tensor_tfm(input_im).unsqueeze(0).to("cuda")*2-1) # Note scaling
  return 0.18215 * latent.latent_dist.sample() # or .mean or .sample

for phase in ['train', 'dev', 'test']:
    current_data = dataset_dict["Brain2Image"][phase].load_data()
    while not current_data["reset"]:
        eeg, labels = current_data["data"], current_data["labels"]
        if labels[0] not in latent_dict:
            latent_dict[labels[0]] = pil_to_latent(vae, Image.open(f"./data/Brain2Image/imageNet_images/{labels[0].split('_')[0]}/{labels[0]}.JPEG").convert("RGB").resize((512, 512)))
        current_data = dataset_dict["Brain2Image"][phase].load_data()
#     print(f"{phase} done.")

import pickle

with open("./data/Brain2Image/latent_dict.pkl", "wb") as f:
    pickle.dump(latent_dict, f)
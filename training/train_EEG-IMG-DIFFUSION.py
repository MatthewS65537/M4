import sys
import time

sys.path.append("./models")
sys.path.append("./training")
sys.path.append("./testing")
sys.path.append("./utils")
sys.path.append("./ZuCo")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from master_init import *
from DSG import *
from data import *
from dataloader import *
from count_params import *

def train_one_epoch(dataloader, model, optimizer, criterion, device="cuda", device_ids=None, staging_device=None):
    if staging_device==None:
        staging_device = f"cuda:{device_ids[0]}" if device_ids == None else "cuda"
    results = {}
    for phase in ['train', 'dev']:
        if phase == 'train':
            model.train()    # Set model to training mode
        else:
            model.eval()     # Set model to evaluate mode

        running_loss = 0.0
        tot_cnt = 0

        # Iterate over data.
        current_data = dataloader[phase].load_data()
        while not current_data["reset"]:
            eeg, image_pixels = current_data["data"]
            latents = vae.encode(image_pixels).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

            noise = torch.randn_like(latents)
            bsz = latents.shape[0]

            timesteps = torch.randint(0, 30, (bsz,), device=latents.device)
            timesteps = timesteps.long()

            noisy_latents = latents + noise
            args_dict = {
                "input_data_batch" : eeg,
                "noisy_latents" : noisy_latents,
                "train" : True
            }
            model_pred = model("EEG-IMG-BRAIN2IMAGE")
            loss = nn.MSELoss()(model_pred, noise)
            loss.backward()
            optimizer.step()

            # # statistics
            running_loss += loss.item() * bsz
            tot_cnt += bsz

            current_data = dataloader[phase].load_data()
        
        epoch_loss = running_loss / tot_cnt
        results[phase] = epoch_loss
        print('{} Loss: {:.4f}'.format(phase, epoch_loss))
    return results

if __name__ == "__main__":
    # config = get_config() Implement later

    config = {
        "device": "cuda:0",
        "device_ids": [0]
    }

    device=config["device"]
    device_ids=config["device_ids"]

    model = INITIALIZE_MODEL(device=device, device_ids=device_ids).to(device)
    dataset_dict = INITIALIZE_DATALOADERS(
        keys=["Brain2Image"],
        bsz=[1]
    )

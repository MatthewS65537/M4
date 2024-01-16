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

import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# Import Pretrain Libraries (transformers + diffusers)
from transformers import BartTokenizer
from diffusers import AutoencoderKL

# Parallel Helper
from parallel import DataParallelModel, DataParallelCriterion

# Import Our Own Functions
from master_init import *
from DSG import *

from count_params import count_params

# Import Tasks
# import EEG_TEXT_BART
# import EEG_TEXT_BART_SENTIMENT
import EEG_IMG_DIFFUSION
# import EEG_IMG_CLASSIFICATION
# import PRETRAIN_EEG_IMG_CLIP_MATCHING
# import PRETRAIN_EEG_TEXT_CLIP_MATCHING
# import PRETRAIN_EEG_IMG_UNET
# import PRETRAIN_EEG_TEXT_UNET

import copy
from tqdm import tqdm

def train_layer(model, param_dict, args_dict, dataset_dict, writers):
    lr = param_dict["lr"]
    beta1 = param_dict["beta1"]
    beta2 = param_dict["beta2"]
    eps = param_dict["eps"]
    weight_decay = param_dict["weight_decay"]
    gamma = param_dict["gamma"]
    gamma_up_ratio = param_dict["gamma_up_ratio"]
    gamma_up = gamma_up_ratio * 1 / (gamma ** 5)
    
    num_epochs = args_dict["num_epochs"]
    vae = args_dict["vae"]
    noise_scheduler = args_dict["noise_scheduler"]
    bsz = args_dict["bsz"]
    device = args_dict["device"]
    device_ids = args_dict["device_ids"]
    staging = args_dict["staging_device"]
    
    train_writer = writers["train"]
    dev_writer = writers["dev"]
    
                
    optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(beta1, beta2), eps=eps, weight_decay=weight_decay)
    scheduler1 = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    scheduler2 = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4,9,14,19,24,29,34,39,44], gamma=gamma_up)
    
    for epoch in tqdm(range(num_epochs)):
        start = time.time()
        args_dict = {
            "model" : model,
            "dataloader" : dataset_dict["Brain2Image"],
            "optimizer" : optimizer,
            "criterion" : nn.MSELoss(),
            "device" : device,
            "device_ids" : device_ids,
            "staging_device" : staging_device,
            "vae" : vae,
            "noise_scheduler" : noise_scheduler,
            "bsz" : bsz
        }
        results = EEG_IMG_DIFFUSION.train(args_dict, using_non_pytorch_parallel=use_non_pytorch_parallel)
        model = results["model"]
        print(f"TRAIN: {results['train_loss']} DEV: {results['dev_loss']} TIME: {time.time() - start:.2f} SECONDS")
        train_writer.add_scalar(f"EEG-IMG-DIFFUSION Loss", results['train_loss'], epoch)
        dev_writer.add_scalar(f"EEG-IMG-DIFFUSION Loss", results['dev_loss'], epoch)
        scheduler1.step()
        scheduler2.step()
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"./checkpoints/Layerwise/MMMM_DIFFUSION-ONLY-NOLORA_{epoch}.pt")
        
    return results

if __name__ == "__main__":
    print(f"[INFO] FETCHING CONFIGURATIONS.")
    config = {
        "device" : "cuda",
        "device_ids" : [0,1,2,3],
        "staging_device" : "cuda",
        "num_epochs" : 50,
        "use_non_pytorch_parallel" : False,
        "test_run" : False,
        "live_evaluate" : True,
        "eval_interval" : 1,
    }

    device = config["device"]
    device_ids = config["device_ids"]
    staging_device = config["staging_device"]
    num_epochs = config["num_epochs"]
    use_non_pytorch_parallel = config["use_non_pytorch_parallel"]
    test_run = config["test_run"]
    live_evaluate = config["live_evaluate"]
    eval_interval = config["eval_interval"]

    print(f"[INFO] FINISHED CONFIGURATIONS.")

    print(f"[INFO] INITIALIZING MODEL.")
    model = INITIALIZE_MODEL(device=None, device_ids=device_ids, dtype=torch.float32)
    if use_non_pytorch_parallel:
        model = DataParallelModel(model, device_ids=device_ids).to(device)
    else:
        model = nn.DataParallel(model, device_ids=device_ids).to(device)
    state_dict = torch.load(f"./tune_checkpoints/BEST-BART.pt")
    model.load_state_dict(state_dict, strict=False)

    for name, param in model.named_parameters():
        param.requires_grad=True
        if ("eeg_encoder" in name) or ("emb_unet" in name) or ("CLIP_text_encoder" in name):
            param.requires_grad=False
            continue
        if "branches" in name:
            if "EEG-TEXT-BART.body" in name:
                if not (("lora" in name) or ("encoder.layers.0" in name) or ("embed_positions" in name) or ("shared" in name)):
                    param.requires_grad=False
                    continue
            if "EEG-IMG-DIFFUSION.body" in name:
                if "lora" in name:
                    print("LORA ACTIVE")
#                 if not (("lora" in name) or ('down_blocks' in name) or ('conv_in' in name) or ('time_embedding' in name)):
                if not (('down_blocks' in name) or ('conv_in' in name) or ('time_embedding' in name)):
#                 if not "lora" in name:
                    param.requires_grad=False
                    continue


    dataset_dict = INITIALIZE_DATALOADERS(
        keys=["Brain2Image"],
        bsz=[1],
        dev_bsz=[1]
    )

    print(f"[INFO] LOADING PRETRAINS...")
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").to(dtype=torch.float32)
    vae.requires_grad_(False)
    
    param_dict = {
        "lr" : 3e-5,
        "beta1" : 0.8,
        "beta2" : 0.9,
        "eps" : 5e-7,
        "weight_decay" : 0.01,
        "gamma" : 0.9,
        "gamma_up_ratio" : 0.3,
    }
    
    noise_scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
    
    args_dict = {
        "num_epochs" : 50,
        "bsz" : 64,
        "vae" : vae,
        "device" : device,
        "device_ids" : device_ids,
        "staging_device" : staging_device,
        "noise_scheduler" : noise_scheduler
    }
    
    train_writer = SummaryWriter(log_dir=f"./logs/train-layerwise-diffusion-only-nolora")
    dev_writer = SummaryWriter(log_dir=f"./logs/dev-layerwise-diffusion-only-nolora")
    writers = {"train" : train_writer, "dev" : dev_writer}
    results = train_layer(model, param_dict, args_dict, dataset_dict, writers)
    model = results["model"]
    torch.save(model.state_dict(), "./checkpoints/Layerwise/MMMM_DIFFUSION-ONLY-NOLORA_FINAL.pt")

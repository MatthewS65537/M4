# Debugged, SLOW
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

from PIL import Image
from torchvision import transforms as tfms

to_tensor_tfm = tfms.ToTensor()

def pil_to_latent(vae, input_im):
  # Single image -> single latent in a batch (so size 1, 4, 64, 64)
  with torch.no_grad():
    latent = vae.encode(to_tensor_tfm(input_im).unsqueeze(0).to("cuda")*2-1) # Note scaling
  return 0.18215 * latent.latent_dist.sample() # or .mean or .sample

def latents_to_pil(vae, latents):
  # bath of latents -> list of images
  latents = (1 / 0.18215) * latents
  with torch.no_grad():
    image = vae.decode(latents)[0]
  image = (image / 2 + 0.5).clamp(0, 1)
  image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
  images = (image * 255).round().astype("uint8")
  pil_images = [Image.fromarray(image) for image in images]
  return pil_images

def train(args_dict, using_non_pytorch_parallel=False):
    dataloader = args_dict["dataloader"]
    model = args_dict["model"]
    optimizer = args_dict["optimizer"]
    criterion = args_dict["criterion"]
    device = args_dict["device"] if "device" in args_dict else "cuda"
    vae = args_dict["vae"]
    device_ids = args_dict["device_ids"] if "device_ids" in args_dict else None
    staging_device = args_dict["staging_device"] if "staging_device" in args_dict else None
    latent_dict = args_dict["latent_dict"]
    del args_dict

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
            eeg, labels = current_data["data"], current_data["labels"]
            latents = [pil_to_latent(vae, Image.open(f"./data/Brain2Image/imageNet_images/{label.split('_')[0]}/{label}.JPEG").convert("RGB")) for label in labels]
            for i in range(len(latents)):
                latent_dict[labels[i]] = latents[i]
                
            if not len(latents) == 1:
                print("[WARNING] BSZ NOT 1. AUTOMATICALLY DROPPING ALL BUT FIRST IN BATCH.")
            latents = latents[0]

            noise = torch.randn_like(latents)
            bsz = latents.shape[0]

            timesteps = torch.randint(0, 30, (bsz,), device=latents.device)
            timesteps = timesteps.long()

            noisy_latents = latents + noise

            optimizer.zero_grad()

            args_dict = {
                "input_data_batch" : eeg.to(device, dtype=torch.float32),
                "pool_result" : True,
                "noisy_latents" : noisy_latents.to(device, dtype=torch.float32),
                "timesteps" : timesteps,
                "train" : True
            }
            model_pred = model("EEG-IMG-DIFFUSION", args_dict)
            
            loss = criterion(model_pred, noise)
            loss.backward()
            optimizer.step()
            
            # # statistics
            running_loss += loss.item() * bsz
            tot_cnt += bsz

            current_data = dataloader[phase].load_data()

        epoch_loss = running_loss / tot_cnt
        results[phase] = epoch_loss
        print('{} Loss: {:.4f}'.format(phase, epoch_loss))

        results[f"{phase}_loss"] = epoch_loss
    results["model"] = model
    results["latent_dict"] = latent_dict
    return results

from master_init import *
from DSG import *

if __name__ == "__main__":
    config = {
        "device" : "cuda:0",
        "device_ids" : [0]
    }
    device = config["device"]
    device_ids = config["device_ids"]

    # model = INITIALIZE_MODEL(device=device, device_ids=device_ids).to(device)
    model = INITIALIZE_MODEL(device=device, device_ids=device_ids).to(device, dtype=torch.float32)

    dataset_dict = INITIALIZE_DATALOADERS(
        keys=["Brain2Image"],
        bsz=[1]
    )

    dsg_tasks = DSGTasks()
    dsg_tasks.add_task(
        DSGTask(
            task_name="EEG-IMG-CLASSIFICATION",
            dataset_tag="Brain2Image",
            criterion=nn.CrossEntropyLoss(), # CE Loss for Tenary Sentiment
            optimizer=optim.Adam,
            learning_rate=5e-3,
            converge_lim=2,
            converge_threshold=0.005,
            div_threshold=0.01
            )
        )

    dataloader = dataset_dict["Brain2Image"]
    model = model
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    criterion = nn.MSELoss()
    device = "cuda"
    device_ids = None
    staging_device = None

    if staging_device == None:
        if device_ids == None:
            staging_device = "cuda"
        else:
            staging_device = f"cuda:{device_ids[0]}"

    from diffusers import AutoencoderKL

    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").to(device, dtype=torch.float16)

    num_epochs = 10
    for epoch_num in range(num_epochs):
        args_dict = {
            "model" : model,
            "dataloader" : dataloader,
            "optimizer" : optimizer,
            "criterion" : criterion,
            "device" : device,
            "device_ids" : device_ids,
            "staging_device" : staging_device,
            "vae" : vae
        }
        results = train(args_dict)
        model = results["model"]
        print(i, results["train_loss"], results["dev_loss"])


    
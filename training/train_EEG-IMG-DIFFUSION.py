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
        # while not current_data["reset"]:
            # input_embeddings, seq_len, input_masks, input_mask_invert, target_ids, target_mask, sentiment_labels, sent_level_EEG = current_data["data"]
                
            # input_embeddings_batch = input_embeddings.to(staging_device).float()
            # input_masks_batch = input_masks.to(staging_device)
            # input_mask_invert_batch = input_mask_invert.to(staging_device)
            # target_ids_batch = target_ids.to(staging_device)
            
            # """replace padding ids in target_ids with -100"""
            # target_ids_batch[target_ids_batch == tokenizer.pad_token_id] = -100 
        
            # optimizer.zero_grad()

            # args_dict = {
            #     "input_data_batch" : input_embeddings_batch,
            #     "input_masks_batch" : input_masks_batch,
            #     "input_masks_invert" : input_mask_invert_batch,
            #     "target_ids_batch" : target_ids_batch
            #     }
            
            # outputs = model(args_dict)
            # loss = criterion(outputs, target_ids_batch)
            # loss.backward()
            # optimizer.step()

            # # statistics
            # running_loss += loss.item() * input_embeddings_batch.size(0)
            # tot_cnt += input_embeddings_batch.size(0)

            # current_data = dataloader[phase].load_data()
        
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

    
import sys
import time

sys.path.append("./models")
sys.path.append("./training")
sys.path.append("./testing")
sys.path.append("./utils")
sys.path.append("./ZuCo")

from master_init import *
from DSG import *
from load_data import *
from data import *
from dataloader import *
from count_params import *

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

def evaluate(args_dict, using_non_pytorch_parallel=False):
    dataloader = args_dict["dataloader"]
    model = args_dict["model"]
    criterion = args_dict["criterion"]
    symmetric_KL = lambda a, b : 0.5 * (criterion(F.log_softmax(a, dim=1), F.softmax(b, dim=1)) + criterion(F.log_softmax(b, dim=1), F.softmax(a, dim=1)))
    device = args_dict["device"] if "device" in args_dict else "cuda"
    device_ids = args_dict["device_ids"] if "device_ids" in args_dict else None
    staging_device = args_dict["staging_device"] if "staging_device" in args_dict else None
    bsz = args_dict["bsz"] if "bsz" in args_dict else 64
    temperature = args_dict["temperature"] if "temperature" in args_dict else 0.04

    if staging_device==None:
        staging_device = f"cuda:{device_ids[0]}" if device_ids == None else "cuda"
    results = {}
    for phase in ['test']:
        dataloader[phase].set_bsz(bsz)
        dataloader[phase].reset()
        model.eval()     # Set model to evaluate mode

        running_loss = 0.0
        tot_cnt = 0
        
        correct = 0
        tot = 0

        # Iterate over data.
        current_data = dataloader[phase].load_data()
        while not current_data["reset"]:
            max_len = max(i.shape[0] for i in current_data["data"])
            eegs = []
            masks = []
            invert_masks = []
            
            if not len(current_data["data"]) == bsz:
                current_data = dataloader[phase].load_data()
                continue

            for i in range(len(current_data["data"])):
                eeg = current_data["data"][i]
                cur_sz = eeg.shape[0]
                mask = torch.cat((torch.ones(cur_sz), torch.zeros(max_len - cur_sz)))
                masks.append(mask)
                invert_masks.append(1 - mask)
                eegs.append(torch.cat((eeg, torch.zeros(max_len - cur_sz, eeg.shape[1]))))

            eeg_batch = torch.stack(eegs)
            masks_batch = torch.stack(masks)
            invert_masks_batch = torch.stack(invert_masks)
            
            args_dict = {
                "input_data_batch" : eeg_batch,
                "input_masks_batch" : masks_batch,
                "input_masks_invert" : invert_masks_batch,
                "pool_result" : True
            }
            
            model.zero_grad()
            
            output = model(
                mode="PRETRAIN-EEG-IMG-CLIP-MATCHING",
                args_dict=args_dict,
                staging_device=staging_device,
            )
            
            lst_targets = current_data["target"]

            target_embed = torch.cat(lst_targets, dim=0).to(dtype=torch.float32)
            output = output.to(dtype=torch.float32)

            target_embeds_pooled = target_embed
            target_pairwise_embeds = torch.mm(target_embeds_pooled, target_embeds_pooled.T)
            output = torch.mm(output, target_embeds_pooled.T)

            loss = symmetric_KL(output, target_pairwise_embeds)

            # Compute stats
            if device_ids == None or len(device_ids) == 1:
                running_loss += loss.item() * len(eegs)
            else:
                running_loss += loss.mean().item() * len(eegs)
            tot_cnt += len(eegs)
            
            for i in range(output.shape[0]):
                current = torch.argmax(output[i])
                if current == i:
                    correct += 1
                tot += 1
            current_data = dataloader[phase].load_data()

        epoch_loss = running_loss / tot_cnt

        results[f"{phase}_loss"] = epoch_loss
        results[f"{phase}_correct"] = correct
        results[f"{phase}_tot"] = tot
        results[f"{phase}_accuracy"] = correct / tot
    return results

if __name__ == "__main__":
    # CKPT_DIR = "./checkpoints/PretrainFinal2"
    # RESULTS_DIR = "./results/PretrainFinal2"
    # MODEL_NAME = "MMMM_39"
    
    CKPT_DIR = "./checkpoints/Pretrain_pe"
    RESULTS_DIR = "./results/PretrainFinal2"
    MODEL_NAME = "MMMM_FINAL"

    
    device="cuda"
    device_ids=[0,1,2,3]
    
    model = INITIALIZE_MODEL(device=device, device_ids=device_ids)
    model = nn.DataParallel(model, device_ids=device_ids).to(device)
    state_dict=torch.load(f"{CKPT_DIR}/{MODEL_NAME}.pt")
    model.load_state_dict(state_dict)
    print("[INFO] Loaded model checkpoint.")
    
    dataloaders = INITIALIZE_DATALOADERS(
        keys=["Brain2Image"],
        bsz=[1],
        dev_bsz=[1]
    )
    
    dataloader=dataloaders["Brain2Image"]
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    
    criterion = nn.KLDivLoss(reduction="batchmean")
    
    args_dict = {
        "dataloader":dataloader,
        "device":device,
        "tokenizer":tokenizer,
        "criterion":criterion,
        "model":model,
        "device_ids":[0],
        "temperature":20
    }
    
    results = {}
    for dev_bsz in [4, 8, 16, 32, 64, 128, 256]:
#         args_dict["dataloader"]["test"] = ZuCo_dataloader["test"].set_bsz(dev_bsz)
        print(dev_bsz)
        args_dict["bsz"] = dev_bsz
        results[dev_bsz] = evaluate(args_dict)
        print(results[dev_bsz])
    import pickle
    with open(f"{RESULTS_DIR}/EIM_SRR_{MODEL_NAME}_pe.pkl", "wb") as f:
        pickle.dump(results, f)
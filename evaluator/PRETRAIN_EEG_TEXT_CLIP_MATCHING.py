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
import torch.optim as optim

from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig

import numpy as np
import pickle


def evaluate(args_dict, using_non_pytorch_parallel=False):
    dataloader = args_dict["dataloader"]
    model = args_dict["model"]
    tokenizer = args_dict["tokenizer"]
    criterion = args_dict["criterion"]
    device = args_dict["device"] if "device" in args_dict else "cuda"
    device_ids = args_dict["device_ids"] if "device_ids" in args_dict else None
    staging_device = args_dict["staging_device"] if "staging_device" in args_dict else None
    temperature = args_dict["temperature"] if "temperature" in args_dict else 20
    symmetric_KL = lambda a, b, t=temperature: 0.5 * (criterion(F.log_softmax(a/t, dim=1), F.softmax(b/t, dim=1)) + criterion(F.log_softmax(b/t, dim=1), F.softmax(a/t, dim=1)))
    dev_bsz = args_dict["dev_bsz"] if "dev_bsz" in args_dict else 64
    bool_eval = args_dict["bool_eval"] if "bool_eval" in args_dict else False
    
    if staging_device==None:
        staging_device = f"cuda:{device_ids[0]}" if device_ids == None else "cuda"
    results = {}
    for phase in ['test']:
        dataloader[phase].set_bsz(dev_bsz)
        dataloader[phase].reset()
        model.eval()

        running_loss = 0.0
        tot_cnt = 0
        
        correct = 0
        tot = 0

        # Iterate over data.
        current_data = dataloader[phase].load_data()
        while not current_data["reset"]:
            input_embeddings, seq_len, input_masks, input_mask_invert, target_ids, target_mask, sentiment_labels, sent_level_EEG = current_data["data"]
#             print(input_embeddings.shape[0])
            if not input_embeddings.shape[0] == dev_bsz:
                current_data = dataloader[phase].load_data()
                continue

            input_embeddings_batch = input_embeddings
            input_masks_batch = input_masks
            input_mask_invert_batch = input_mask_invert
            target_ids_batch = target_ids

            """replace padding ids in target_ids with -100"""
            target_ids_batch[target_ids_batch == tokenizer.pad_token_id] = -100

            model.zero_grad()

            args_dict = {
                "input_data_batch" : input_embeddings_batch.to(staging_device, dtype=torch.float32),
                "input_masks_batch" : input_masks_batch.to(staging_device, dtype=torch.float32),
                "input_masks_invert" : input_mask_invert_batch.to(staging_device, dtype=torch.float32),
                "target_ids_batch" : target_ids_batch.to(staging_device),
                "pool_result" : False
                }

            output = model(
                mode="PRETRAIN-EEG-TEXT-CLIP-MATCHING",
                args_dict=args_dict,
                staging_device=staging_device,
            )

            target_embed = current_data["target"].to(torch.float32)
            target_embeds_pooled = torch.mean(target_embed, dim=1)
            target_pairwise_embeds = torch.mm(target_embeds_pooled, target_embeds_pooled.T)

            output = torch.mean(output, dim=1).to(torch.float32)
            output = torch.mm(output, target_embeds_pooled.T)

            loss = symmetric_KL(output, target_pairwise_embeds)

            # Compute stats
            if device_ids == None or len(device_ids) == 1:
                running_loss += loss.item() * input_embeddings_batch.size()[0]
            else:
                running_loss += loss.mean().item() * input_embeddings_batch.size()[0]
            tot_cnt += input_embeddings_batch.size()[0]
            
            current_data = dataloader[phase].load_data()

            for i in range(output.shape[0]):
                current = torch.argmax(output[i])
                if current == i:
                    correct += 1
                tot += 1

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
    
    CKPT_DIR = "./checkpoints/layerwise_pe"
    RESULTS_DIR = "./results/ablation"
    MODEL_NAME = "MMMM_ALL_FINAL"

    device="cuda"
    device_ids=[0,1,2,3]
    
    model = INITIALIZE_MODEL(device=device, device_ids=device_ids)
    model = nn.DataParallel(model, device_ids=device_ids).to(device)
    state_dict=torch.load(f"{CKPT_DIR}/{MODEL_NAME}.pt")
    model.load_state_dict(state_dict)
    print("[INFO] Loaded model checkpoint.")
    
    dataloaders = INITIALIZE_DATALOADERS(
        keys=["ZuCo-CLIP"],
        bsz=[1],
        dev_bsz=[1]
    )
    
    ZuCo_dataloader=dataloaders["ZuCo-CLIP"]
    BART_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
    
    criterion = nn.KLDivLoss(reduction="batchmean")
    
    args_dict = {
        "dataloader":ZuCo_dataloader,
        "device":device,
        "criterion":criterion,
        "tokenizer":BART_tokenizer,
        "model":model,
        "device_ids":device_ids,
        "temperature":20
    }
    
    results = {}
    for dev_bsz in [4, 8, 16, 32, 64, 128, 256]:
        print(dev_bsz)
        args_dict["dev_bsz"] = dev_bsz
        results[dev_bsz] = evaluate(args_dict)
        print(results[dev_bsz])
    import pickle
    # with open(f"{RESULTS_DIR}/ETM_SRR-{MODEL_NAME}_pe.pkl", "wb") as f:
    #     pickle.dump(results, f)
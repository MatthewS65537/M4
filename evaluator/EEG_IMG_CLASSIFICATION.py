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
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

def calculate_topk_error(output, expected, k=1):

    expected = torch.tensor(expected, device=output.device)
    # tok-k idx
    _, topk = output.topk(k, dim=1)
    correct = topk.eq(expected.view(-1, 1).expand_as(topk))
    error = 1 - correct.any(dim=1).float().mean().item()

    return error


def evaluate(args_dict, using_non_pytorch_parallel=False):
    dataloader = args_dict["dataloader"]
    model = args_dict["model"]
    criterion = args_dict["criterion"]
    device = args_dict["device"] if "device" in args_dict else "cuda"
    device_ids = args_dict["device_ids"] if "device_ids" in args_dict else None
    staging_device = args_dict["staging_device"] if "staging_device" in args_dict else None
    bsz = args_dict["bsz"] if "bsz" in args_dict else 256
    temperature = args_dict["temperature"] if "temperature" in args_dict else 0.04

    if staging_device==None:
        staging_device = f"cuda:{device_ids[0]}" if device_ids == None else "cuda"
    results = {}
    for phase in ['test']:
        dataloader[phase].reset()
        dataloader[phase].set_bsz(bsz)
        model.eval()     # Set model to evaluate mode

        running_loss = 0.0
        tot_cnt = 0
        confusion_matrix = np.zeros((40, 40))

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
                "pool_result" : True,
                "temperature" : temperature
            }
            
            model.zero_grad()
            
            output = model(
                mode="EEG-IMG-BRAIN2IMAGE-CLASSIFICATION",
                args_dict=args_dict,
                staging_device=staging_device
                )
            
            expected = current_data["classes"]
            for i in range(output.shape[0]):
                confusion_matrix[expected[i]][np.argmax(output[i].cpu().detach().numpy())] += 1
            
            target = torch.zeros(len(eegs), 40).to(device, dtype=torch.float32)
            for i in range(len(expected)):
                target[i][expected[i]] = 1.0
            
            loss = criterion(output.to(dtype=torch.float32), target.to(dtype=torch.float32))

            # Compute stats
            if device_ids == None:
                running_loss += loss.item() * len(eegs)
            else:
                running_loss += loss.mean().item() * len(eegs)
            tot_cnt += len(eegs)
            current_data = dataloader[phase].load_data()

        epoch_loss = running_loss / tot_cnt

        results[f"{phase}_loss"] = epoch_loss
        results[f"{phase}_confusion_matrix"] = confusion_matrix
        tp = np.array([confusion_matrix[i][i] for i in range(40)])
        # 计算 FP 和 FN
        fp = np.sum(confusion_matrix, axis=0) - tp  # 列的总和减去 TP
        fn = np.sum(confusion_matrix, axis=1) - tp  # 行的总和减去 TP
        # Calculate TN
        # tn = np.zeros(40)
        # for i in range(40):
        #     temp_matrix = np.delete(confusion_matrix, i, 0)  # Delete row
        #     temp_matrix = np.delete(temp_matrix, i, 1)      # Delete column
        #     tn[i] = np.sum(temp_matrix)

        # 计算整体指标（微平均和宏平均）
        micro_avg_precision = tp.sum() / (tp.sum() + fp.sum())
        micro_avg_recall = tp.sum() / (tp.sum() + fn.sum())
        micro_avg_f1 = 2 * (micro_avg_precision * micro_avg_recall) / (micro_avg_precision + micro_avg_recall)

        macro_avg_precision = precision.mean()
        macro_avg_recall = recall.mean()
        macro_avg_f1 = f1.mean()
        
        results[f"{phase}_TP"] = tp
        results[f"{phase}_FP"] = fp
        # results[f"{phase}_TN"] = tn
        results[f"{phase}_FN"] = fn
        results[f"{phase}_precision"] = precision = tp / (tp + fp)
        results[f"{phase}_recall"] = recall = tp / (tp + fn)
        results[f"{phase}_f1"] = f1 = 2 * (precision * recall) / (precision + recall)
        results[f"{phase}_macro_avg_precision"] = macro_avg_precision
        results[f"{phase}_macro_avg_recall"] = macro_avg_recall
        results[f"{phase}_macro_avg_f1"] = macro_avg_f1
        results[f"{phase}_micro_avg_precision"] = micro_avg_precision
        results[f"{phase}_micro_avg_recall"] = micro_avg_recall
        results[f"{phase}_micro_avg_f1"] = micro_avg_f1
        # Top-K errors
        top1_error = calculate_topk_error(output, expected, k=1)
        top5_error = calculate_topk_error(output, expected, k=5)
        top10_error = calculate_topk_error(output, expected, k=10)

        # print Top-K
        print(f"Top-1 Error: {top1_error}")
        print(f"Top-5 Error: {top5_error}")
        print(f"Top-10 Error: {top10_error}")
    return results

if __name__ == "__main__":
    # CKPT_DIR = "./checkpoints/layerwise"
    # RESULTS_DIR = "./results/layerwise-dropout--0-5-FINAL"
    # MODEL_NAME = "MMMM_SENT-ONLY_FINAL-DROPOUT--0-5-FINAL"
    CKPT_DIR = "./checkpoints/DSG"
    RESULTS_DIR = "./tune_results/BART"
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
    
    criterion = nn.CrossEntropyLoss()
    
    args_dict = {
        "dataloader":dataloader,
        "device":device,
        "tokenizer":tokenizer,
        "criterion":criterion,
        "model":model,
        "device_ids":[0],
        "temperature":0.04
    }
    
    results = evaluate(args_dict)
    print(results)
    with open(f"{RESULTS_DIR}/IMG_CLASSIFICATION_DSG_pe.pkl", "wb") as f:
        pickle.dump(results, f)
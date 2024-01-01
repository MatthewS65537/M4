# Debugged
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

def train(args_dict, using_non_pytorch_parallel=False):
    dataloader = args_dict["dataloader"]
    model = args_dict["model"]
    optimizer = args_dict["optimizer"]
    criterion = args_dict["criterion"]
    device = args_dict["device"] if "device" in args_dict else "cuda"
    device_ids = args_dict["device_ids"] if "device_ids" in args_dict else None
    staging_device = args_dict["staging_device"] if "staging_device" in args_dict else None
    bsz = args_dict["bsz"] if "bsz" in args_dict else 256
    temperature = args_dict["temperature"] if "temperature" in args_dict else 20

    if staging_device==None:
        staging_device = f"cuda:{device_ids[0]}" if device_ids == None else "cuda"
    results = {}
    for phase in ['train', 'dev']:
        if phase == 'train':
            dataloader[phase].set_bsz(bsz)
            model.train()    # Set model to training mode
        else:
            dataloader[phase].set_bsz(bsz)
            model.eval()     # Set model to evaluate mode

        running_loss = 0.0
        tot_cnt = 0

        # Iterate over data.
        current_data = dataloader[phase].load_data()
        while not current_data["reset"]:
            max_len = max(i.shape[0] for i in current_data["data"])
            eegs = []
            masks = []
            invert_masks = []

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
            
            output = model(
                mode="EEG-IMG-BRAIN2IMAGE-CLASSIFICATION",
                args_dict=args_dict,
                staging_device=staging_device
                )
            
            expected = current_data["classes"]
            
            target = torch.zeros(len(eegs), 40).to(device, dtype=torch.float32)
            for i in range(len(expected)):
                target[i][expected[i]] = 1.0
            
            loss = criterion(output.to(dtype=torch.float32), target.to(dtype=torch.float32))

            # Backward + Optimize only if in training phase
            if phase == 'train':
                if device_ids == None:
                    loss.backward()
                    optimizer.step()
                else:
                    loss.mean().backward()
                    optimizer.step()

            # Compute stats
            if device_ids == None:
                running_loss += loss.item() * len(eegs)
            else:
                running_loss += loss.mean().item() * len(eegs)
            tot_cnt += len(eegs)
            current_data = dataloader[phase].load_data()

        epoch_loss = running_loss / tot_cnt

        results[f"{phase}_loss"] = epoch_loss
    results["model"] = model
    return results

def evaluate(args_dict):
    dataloader = args_dict["dataloader"]
    model = args_dict["model"]
    device = args_dict["device"] if "device" in args_dict else "cuda"
    device_ids = args_dict["device_ids"] if "device_ids" in args_dict else None
    staging_device = args_dict["staging_device"] if "staging_device" in args_dict else None
    if staging_device==None:
        staging_device = f"cuda:{device_ids[0]}" if device_ids == None else "cuda"
    results = {}
    for phase in ['train', 'dev']:
        if phase == 'train':
            model.train()    # Set model to training mode
        else:
            model.eval()     # Set model to evaluate mode

        tot_cnt = 0
        confusion_matrix = np.zeros((40, 40))

        # Iterate over data.
        current_data = dataloader[phase].load_data()
        while not current_data["reset"]:
            eegs = current_data["data"]
            expected = current_data["classes"]

            model.zero_grad()

            args_dict = {
                "input_data_batch" : eegs.to(device),
                "pool_result" : True
                }

            output = model(
                mode="EEG-IMG-BRAIN2IMAGE-CLASSIFICATION",
                args_dict=args_dict,
                staging_device=staging_device
                )

            for i in range(output.shape[0]):
                confusion_matrix[expected[i]][np.argmax(output[i].cpu().detach().numpy())] += 1
            
            tot_cnt += eegs.size()[0]
            current_data = dataloader[phase].load_data()

        results[f"{phase}_confusion_matrix"] = confusion_matrix
        tp = np.array([confusion_matrix[i][i] for i in range(40)])
        fp = np.array([confusion_matrix[i,:].sum() - confusion_matrix[i][i] for i in range(40)]) - tp
        fn = np.array([confusion_matrix[:,i].sum() - confusion_matrix[i][i] for i in range(40)]) - tp
        tn = np.array([tot_cnt - tp[i] - fp[i] - fn[i] for i in range(40)])
        results[f"{phase}_TP"] = tp
        results[f"{phase}_FP"] = fp
        results[f"{phase}_TN"] = tn
        results[f"{phase}_FN"] = fn
        results[f"{phase}_accuracy"] = accuracy = (tp + tn) / tot_cnt
        results[f"{phase}_precision"] = precision = tp / (tp + fp)
        results[f"{phase}_recall"] = recall = tp / (tp + fn)
        results[f"{phase}_f1"] = f1 = 2 * precision * recall / (precision + recall)
    model.zero_grad()
        
    return results
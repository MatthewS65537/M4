# Debugged
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

def train(args_dict, using_non_pytorch_parallel=False):
    dataloader = args_dict["dataloader"]
    model = args_dict["model"]
    optimizer = args_dict["optimizer"]
    tokenizer = args_dict["tokenizer"]
    criterion = args_dict["criterion"]
    device = args_dict["device"] if "device" in args_dict else "cuda"
    device_ids = args_dict["device_ids"] if "device_ids" in args_dict else None
    staging_device = args_dict["staging_device"] if "staging_device" in args_dict else None

    if staging_device==None:
        staging_device = f"cuda:{device_ids[0]}" if not device_ids == None else "cuda"
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
            input_embeddings, seq_len, input_masks, input_mask_invert, target_ids, target_mask, sentiment_labels, sent_level_EEG = current_data["data"]

            good = []
            for i in range(len(sentiment_labels)):
                if not sentiment_labels[i] == -100:
                    good.append(i)
            
            if len(good) == 0:
                current_data = dataloader[phase].load_data()
                continue
            
            input_embeddings = input_embeddings[good]
            input_masks = input_masks[good]
            input_mask_invert = input_mask_invert[good]
            target_ids = target_ids[good]
            sentiment_labels = sentiment_labels[good]

            target = torch.zeros(input_embeddings.shape[0], 3).to(device)
            for i in range(input_embeddings.shape[0]):
                target[i][sentiment_labels[i]] = 1.0

            input_embeddings_batch = input_embeddings.to(staging_device, dtype=torch.float32)
            input_masks_batch = input_masks.to(staging_device, dtype=torch.float32)
            input_mask_invert_batch = input_mask_invert.to(staging_device, dtype=torch.float32)
            target_ids_batch = target_ids.to(staging_device)

            """replace padding ids in target_ids with -100"""
            target_ids_batch[target_ids_batch == tokenizer.pad_token_id] = -100

            optimizer.zero_grad()

            args_dict = {
                "input_data_batch" : input_embeddings_batch,
                "input_masks_batch" : input_masks_batch,
                "input_masks_invert" : input_mask_invert_batch,
                "target_ids_batch" : target_ids_batch,
                "pool_result" : True
                }

            output = model(
                mode="EEG-TEXT-BART-SENTIMENT",
                args_dict=args_dict,
                staging_device=staging_device
                )

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
                running_loss += loss.item() * input_embeddings_batch.size()[0]
            else:
                running_loss += loss.mean().item() * input_embeddings_batch.size()[0]
            tot_cnt += input_embeddings_batch.size()[0]
            current_data = dataloader[phase].load_data()

        epoch_loss = running_loss / tot_cnt

        results[f"{phase}_loss"] = epoch_loss
    results["model"] = model
    return results

def evaluate(args_dict):
    dataloader = args_dict["dataloader"]
    model = args_dict["model"]
    tokenizer = args_dict["tokenizer"]
    device = args_dict["device"] if "device" in args_dict else "cuda"
    device_ids = args_dict["device_ids"] if "device_ids" in args_dict else None
    staging_device = args_dict["staging_device"] if "staging_device" in args_dict else None

    if staging_device==None:
        staging_device = f"cuda:{device_ids[0]}" if not device_ids == None else "cuda"
    results = {}
    for phase in ['train', 'dev']:
        confusion_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        model.eval()     # Set model to evaluate mode

        # Iterate over data.
        current_data = dataloader[phase].load_data()
        while not current_data["reset"]:
            input_embeddings, seq_len, input_masks, input_mask_invert, target_ids, target_mask, sentiment_labels, sent_level_EEG = current_data["data"]

            good = []
            for i in range(len(sentiment_labels)):
                if not sentiment_labels[i] == -100:
                    good.append(i)
            
            if len(good) == 0:
                current_data = dataloader[phase].load_data()
                continue
            
            input_embeddings = input_embeddings[good]
            input_masks = input_masks[good]
            input_mask_invert = input_mask_invert[good]
            target_ids = target_ids[good]
            sentiment_labels = sentiment_labels[good]

            target = torch.zeros(input_embeddings.shape[0], 3).to(device)
            for i in range(input_embeddings.shape[0]):
                target[i][sentiment_labels[i]] = 1.0

            input_embeddings_batch = input_embeddings.to(staging_device).float()
            input_masks_batch = input_masks.to(staging_device)
            input_mask_invert_batch = input_mask_invert.to(staging_device)
            target_ids_batch = target_ids.to(staging_device)

            """replace padding ids in target_ids with -100"""
            target_ids_batch[target_ids_batch == tokenizer.pad_token_id] = -100

            model.zero_grad()

            args_dict = {
                "input_data_batch" : input_embeddings_batch,
                "input_masks_batch" : input_masks_batch,
                "input_masks_invert" : input_mask_invert_batch,
                "target_ids_batch" : target_ids_batch,
                "pool_result" : True
                }

            output = model(
                mode="EEG-TEXT-BART-SENTIMENT",
                args_dict=args_dict,
                staging_device=staging_device
                )
            
            for i in range(input_embeddings.shape[0]):
                confusion_matrix[sentiment_labels[i]][torch.argmax(output[i])] += 1
            
            current_data = dataloader[phase].load_data()
        
        confusion_matrix = np.array(confusion_matrix)
        results[f"{phase}_confusion_matrix"] = confusion_matrix
        tp = np.array([confusion_matrix[0][0], confusion_matrix[1][1], confusion_matrix[2][2]])
        fp = np.array([confusion_matrix[1][0] + confusion_matrix[2][0],
        confusion_matrix[0][1] + confusion_matrix[2][1],
        confusion_matrix[0][2] + confusion_matrix[1][2]])
        fn = np.array([confusion_matrix[0][1] + confusion_matrix[0][2],
        confusion_matrix[1][0] + confusion_matrix[1][2],
        confusion_matrix[2][0] + confusion_matrix[2][1]])
        tn = np.array([confusion_matrix[1][1] + confusion_matrix[1][2] + confusion_matrix[2][1] + confusion_matrix[2][2],
        confusion_matrix[0][0] + confusion_matrix[0][2] + confusion_matrix[2][0] + confusion_matrix[2][2],
        confusion_matrix[0][0] + confusion_matrix[0][1] + confusion_matrix[1][0] + confusion_matrix[1][1]])
        results[f"{phase}_TP"] = tp
        results[f"{phase}_FP"] = fp
        results[f"{phase}_TN"] = tn
        results[f"{phase}_FN"] = fn
        results[f"{phase}_accuracy"] = accuracy = (tp + tn) / (tp + tn + fp + fn)
        results[f"{phase}_precision"] = precision = tp / (tp + fp)
        results[f"{phase}_recall"] = recall = tp / (tp + fn)
        results[f"{phase}_f1"] = f1 = 2 * (precision * recall) / (precision + recall)
    model.zero_grad()
    return results
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
    temperature = args_dict["temperature"] if "temperature" in args_dict else 20
    symmetric_KL = lambda a, b, t=temperature: 0.5 * (criterion(F.log_softmax(a/t, dim=1), F.softmax(b/t, dim=1)) + criterion(F.log_softmax(b/t, dim=1), F.softmax(a/t, dim=1)))
    dev_bsz = args_dict["dev_bsz"] if "dev_bsz" in args_dict else 64
    bool_eval = args_dict["bool_eval"] if "bool_eval" in args_dict else False
    
    if staging_device==None:
        staging_device = f"cuda:{device_ids[0]}" if device_ids == None else "cuda"
    results = {}
    for phase in ['train', 'dev']:
        if phase == 'train':
            dataloader[phase].reset()
            model.train()    # Set model to training mode
        else:
            dataloader[phase].set_bsz(dev_bsz)
            dataloader[phase].reset()
            model.eval()     # Set model to evaluate mode

        running_loss = 0.0
        tot_cnt = 0
        
        correct = 0
        tot = 0

        # Iterate over data.
        current_data = dataloader[phase].load_data()
        while not current_data["reset"]:
            input_embeddings, seq_len, input_masks, input_mask_invert, target_ids, target_mask, sentiment_labels, sent_level_EEG = current_data["data"]

            input_embeddings_batch = input_embeddings
            input_masks_batch = input_masks
            input_mask_invert_batch = input_mask_invert
            target_ids_batch = target_ids

            """replace padding ids in target_ids with -100"""
            target_ids_batch[target_ids_batch == tokenizer.pad_token_id] = -100

            optimizer.zero_grad()

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

            # Normalize for Stable Softmax
            target_pairwise_embeds = torch.div(torch.max(target_pairwise_embeds, dim=1).values.unsqueeze(1))
            output = torch.div(torch.max(output, dim=1).values.unsqueeze(1))
            
            loss = symmetric_KL(output, target_pairwise_embeds)
            
            # Backward + Optimize only if in training phase
            if phase == 'train':
                if device_ids == None:
                    loss.backward()
                    optimizer.step()
                else:
                    loss.mean().backward()
                    optimizer.step()

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
        
        if bool_eval:
            results[f"{phase}_accuracy"] = correct / tot
    results["model"] = model
    return results

# def evaluate(args_dict):
#     dataloader = args_dict["dataloader"]
#     model = args_dict["model"]
#     tokenizer = args_dict["tokenizer"]
#     device = args_dict["device"] if "device" in args_dict else "cuda"
#     device_ids = args_dict["device_ids"] if "device_ids" in args_dict else None
#     staging_device = args_dict["staging_device"] if "staging_device" in args_dict else None
#     dev_bsz = args_dict["dev_bsz"] if "dev_bsz" in args_dict else 64

#     if staging_device==None:
#         staging_device = f"cuda:{device_ids[0]}" if device_ids == None else "cuda"
#     results = {}
#     for phase in ['train', 'dev']:
#         if phase == 'train':
#             model.train()    # Set model to training mode
#         else:
#             dataloader[phase].set_bsz(dev_bsz)
#             model.eval()     # Set model to evaluate mode

#         correct = 0
#         tot = 0

#         # Iterate over data.
#         current_data = dataloader[phase].load_data()
#         while not current_data["reset"]:
#             input_embeddings, seq_len, input_masks, input_mask_invert, target_ids, target_mask, sentiment_labels, sent_level_EEG = current_data["data"]

#             input_embeddings_batch = input_embeddings
#             input_masks_batch = input_masks
#             input_mask_invert_batch = input_mask_invert
#             target_ids_batch = target_ids

#             """replace padding ids in target_ids with -100"""
#             target_ids_batch[target_ids_batch == tokenizer.pad_token_id] = -100

#             model.zero_grad()

#             args_dict = {
#                 "input_data_batch" : input_embeddings_batch.to(dtype=torch.float32),
#                 "input_masks_batch" : input_masks_batch.to(dtype=torch.float32),
#                 "input_masks_invert" : input_mask_invert_batch.to(dtype=torch.float32),
#                 "target_ids_batch" : target_ids_batch,
#                 "pool_result" : False
#                 }

#             output = model(
#                 mode="PRETRAIN-EEG-TEXT-CLIP-MATCHING",
#                 args_dict=args_dict,
#                 staging_device=staging_device,
#             )

#             target_embed = current_data["target"].to(torch.float32)
#             target_embeds_pooled = torch.mean(target_embed, dim=1)
#             target_pairwise_embeds = torch.mm(target_embeds_pooled, target_embeds_pooled.T)

#             output = torch.mean(output, dim=1).to(torch.float32)
#             output = torch.mm(output, target_embeds_pooled.T)

#             for i in range(output.shape[0]):
#                 current = torch.argmax(output[i])
#                 if current == i:
#                     correct += 1
#                 tot += 1

#             current_data = dataloader[phase].load_data()
#             results[f"{phase}_accuracy"] = correct / tot
#     model.zero_grad()
#     return results
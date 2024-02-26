# FINISHED DEBUG
import torch
import torch.nn as nn
import torch.nn.functional as F

def train(args_dict, using_non_pytorch_parallel=False):
    dataloader = args_dict["dataloader"]
    model = args_dict["model"]
    optimizer = args_dict["optimizer"]
    tokenizer = args_dict["tokenizer"]
    device = args_dict["device"] if "device" in args_dict else "cuda"
    device_ids = args_dict["device_ids"] if "device_ids" in args_dict else None
    staging_device = args_dict["staging_device"] if "staging_device" in args_dict else None
    pref_dtype = args_dict["pref_dtype"] if "pref_dtype" in args_dict else torch.float32

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
            input_embeddings, seq_len, input_masks, input_mask_invert, target_ids, target_mask, sentiment_labels, sent_level_EEG = current_data["data"]
                
            input_embeddings_batch = input_embeddings
            input_masks_batch = input_masks
            input_mask_invert_batch = input_mask_invert
            target_ids_batch = target_ids
            
            """replace padding ids in target_ids with -100"""
            target_ids_batch[target_ids_batch == tokenizer.pad_token_id] = -100 
        
            optimizer.zero_grad()

            args_dict = {
                "input_data_batch" : input_embeddings_batch.to(staging_device, dtype=pref_dtype),
                "input_masks_batch" : input_masks_batch.to(staging_device, dtype=pref_dtype),
                "input_masks_invert" : input_mask_invert_batch.to(staging_device, dtype=pref_dtype),
                "target_ids_batch" : target_ids_batch.to(staging_device),
                "pool_result" : False
                }
            
            seq2seqLMoutput = model(
                mode="EEG-TEXT-BART",
                args_dict=args_dict,
                staging_device=staging_device,
#                 debug=True
                )
            
#             print(seq2seqLMoutput)
            
            # Use the BART language modeling loss
            if using_non_pytorch_parallel:
                loss = None
                for LMoutput in seq2seqLMoutput:
                    if loss == None:
                        loss = LMoutput.loss.to(staging_device)
                    else:
                        loss += LMoutput.loss.to(staging_device)
                loss /= len(seq2seqLMoutput)
            else:
                loss = seq2seqLMoutput.loss
            loss.to(dtype=torch.float32)
            
            # Backward + Optimize only if in training phase
            if phase == 'train':
                if device_ids == None:
                    loss.backward()
                    nn.utils.clip_grad_value_(model.parameters(), 15.0)
                    optimizer.step()
                elif using_non_pytorch_parallel:
                    loss.backward()
                    nn.utils.clip_grad_value_(model.parameters(), 15.0)
                    optimizer.step()
                else:
                    loss.mean().backward()
                    nn.utils.clip_grad_value_(model.parameters(), 15.0)
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
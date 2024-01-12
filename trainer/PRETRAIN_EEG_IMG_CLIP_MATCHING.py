import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

def train(args_dict, using_non_pytorch_parallel=False, verbose=False):
    dataloader = args_dict["dataloader"]
    model = args_dict["model"]
    optimizer = args_dict["optimizer"]
    criterion = args_dict["criterion"]
#     softmax_norm = lambda x : x # Dummy
#     softmax_norm = lambda x : F.normalize(x, p=2, dim=1) # Doesn't work
    softmax_norm = lambda x : x - x.max(dim=1).values.unsqueeze(dim=1)

    temperature = args_dict["temperature"] if "temperature" in args_dict else 25
    symmetric_KL = lambda a, b, t=temperature: 0.5 * (criterion(F.log_softmax(a/t, dim=1), F.softmax(b/t, dim=1)) + criterion(F.log_softmax(b/t, dim=1), F.softmax(a/t, dim=1)))
    device = args_dict["device"] if "device" in args_dict else "cuda"
    device_ids = args_dict["device_ids"] if "device_ids" in args_dict else None
    staging_device = args_dict["staging_device"] if "staging_device" in args_dict else None
    bsz = args_dict["bsz"] if "bsz" in args_dict else 64
    bool_eval = args_dict["bool_eval"] if "bool_eval" in args_dict else False
    use_unet = args_dict["use_unet"] if "use_unet" in args_dict else False

    if staging_device==None:
        staging_device = f"cuda:{device_ids[0]}" if device_ids == None else "cuda"
    results = {}
    for phase in ['train', 'dev']:
        if phase == 'train':
            dataloader[phase].set_bsz(bsz)
            dataloader[phase].reset()
            model.train()    # Set model to training mode
        else:
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
                "use_unet" : use_unet
            }
            
            optimizer.zero_grad()
            
            output = model(
                mode="PRETRAIN-EEG-IMG-CLIP-MATCHING",
                args_dict=args_dict,
                staging_device=staging_device,
            )
            
            lst_targets = current_data["target"]
            
#             for i in range(real_bsz):
#                 eegs = current_data["data"]
#                 if current_data["reset"]:
#                     break
                

#                 model.zero_grad()
#                 output = model(
#                     mode="PRETRAIN-EEG-IMG-CLIP-MATCHING",
#                     args_dict=args_dict,
#                     staging_device=staging_device,
#                 )

#                 lst_outputs.append(output)
#                 lst_targets.append(current_data["target"][0])
#                 current_data = dataloader[phase].load_data()

            target_embed = torch.cat(lst_targets, dim=0).to(dtype=torch.float32)
#             output = torch.cat(lst_outputs, dim=0).to(dtype=torch.float32)
            output = output.to(dtype=torch.float32)

            target_embeds_pooled = target_embed
            target_pairwise_embeds = torch.mm(target_embeds_pooled, target_embeds_pooled.T)
            output = torch.mm(output, target_embeds_pooled.T)
            
            # Normalize for Stable Softmax
            loss = symmetric_KL(output, target_pairwise_embeds)
            
#             print(F.softmax(softmax_norm(output)/temperature)[:8,:8])
#             print(F.softmax(softmax_norm(target_pairwise_embeds)/temperature)[:8,:8])
#             break
            
            # Backward + Optimize only if in training phase
            if phase == 'train':
                if device_ids == None:
                    loss.backward()
                    nn.utils.clip_grad_value_(model.parameters(), 15.0)
                    optimizer.step()
                else:
                    loss.mean().backward()
                    nn.utils.clip_grad_value_(model.parameters(), 15.0)
                    optimizer.step()
#                 print(loss.item())
#                 print("Max: ", torch.max(torch.stack([torch.max(torch.tensor([0.0]).to(device) if parameter.grad == None else parameter.grad) for name, parameter in model.named_parameters()])))
#                 nn.utils.clip_grad_value_(model.parameters(), 10.0)
#                 print("Normed Max:", torch.max(torch.stack([torch.max(torch.tensor([0.0]).to(device) if parameter.grad == None else parameter.grad) for name, parameter in model.named_parameters()])))
#                 optimizer.step()

            # Compute stats
            if device_ids == None or len(device_ids) == 1:
                running_loss += loss.item() * len(eegs)
            else:
                running_loss += loss.mean().item() * len(eegs)
            tot_cnt += len(eegs)
            
            if bool_eval:
                for i in range(output.shape[0]):
                    current = torch.argmax(output[i])
                    if current == i:
                        correct += 1
                    tot += 1
            current_data = dataloader[phase].load_data()

        epoch_loss = running_loss / tot_cnt

        results[f"{phase}_loss"] = epoch_loss
        if bool_eval:
            results[f"{phase}_accuracy"] = correct / tot
            if verbose:
                print(f"{phase}: {correct}/{tot}")
    results["model"] = model
    return results

def evaluate(args_dict):
    dataloader = args_dict["dataloader"]
    model = args_dict["model"]
    device = args_dict["device"] if "device" in args_dict else "cuda"
    device_ids = args_dict["device_ids"] if "device_ids" in args_dict else None
    staging_device = args_dict["staging_device"] if "staging_device" in args_dict else None
    real_bsz = args_dict["real_bsz"] if "real_bsz" in args_dict else 8

    if staging_device==None:
        staging_device = f"cuda:{device_ids[0]}" if device_ids == None else "cuda"
    results = {}
    for phase in ['train', 'dev']:
        if phase == 'train':
            model.train()    # Set model to training mode
        else:
            model.eval()     # Set model to evaluate mode

        correct = 0
        tot = 0

        # Iterate over data.
        current_data = dataloader[phase].load_data()
        while not current_data["reset"]:
            lst_targets = []
            lst_outputs = []
            for i in range(real_bsz):
                eegs = current_data["data"]
                if current_data["reset"]:
                    break
                args_dict = {
                    "input_data_batch" : eegs.to(device, dtype=torch.float32),
                    "pool_result" : True
                    }

                model.zero_grad()
                output = model(
                    mode="PRETRAIN-EEG-IMG-CLIP-MATCHING",
                    args_dict=args_dict,
                    staging_device=staging_device,
                )

                lst_outputs.append(output)
                lst_targets.append(current_data["target"][0])
                current_data = dataloader[phase].load_data()

            target_embed = torch.cat(lst_targets, dim=0).to(dtype=torch.float32)
            output = torch.cat(lst_outputs, dim=0).to(dtype=torch.float32)

            model.zero_grad()

            target_embeds_pooled = target_embed
            target_pairwise_embeds = torch.mm(target_embeds_pooled, target_embeds_pooled.T)
            output = torch.mm(output, target_embeds_pooled.T)

            for i in range(output.shape[0]):
                current = torch.argmax(output[i])
                if current == i:
                    correct += 1
                tot += 1

            current_data = dataloader[phase].load_data()
        results[f"{phase}_accuracy"] = correct / tot
    model.zero_grad()
    return results
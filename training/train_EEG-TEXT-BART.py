import sys
import time

sys.path.append("./models")
sys.path.append("./training")
sys.path.append("./testing")
sys.path.append("./utils")
sys.path.append("./ZuCo")

from master_init import *
from DSG import *
from data import *
from dataloader import *
from count_params import *

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

from transformers import BartTokenizer

from argparser import get_config

def train_one_epoch(dataloader, model, optimizer, criterion, tokenizer, device="cuda", device_ids=None, staging_device=None):
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
                
            input_embeddings_batch = input_embeddings.to(staging_device).float()
            input_masks_batch = input_masks.to(staging_device)
            input_mask_invert_batch = input_mask_invert.to(staging_device)
            target_ids_batch = target_ids.to(staging_device)
            
            """replace padding ids in target_ids with -100"""
            target_ids_batch[target_ids_batch == tokenizer.pad_token_id] = -100 
        
            optimizer.zero_grad()

            args_dict = {
                "input_data_batch" : input_embeddings_batch,
                "input_masks_batch" : input_masks_batch,
                "input_masks_invert" : input_mask_invert_batch,
                "target_ids_batch" : target_ids_batch
                }
            
            seq2seqLMoutput = model(
                mode="EEG-TEXT-BART",
                args_dict=args_dict,
                staging_device=staging_device
                )
            
            # Use the BART language modeling loss
            loss = seq2seqLMoutput.loss
            
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

if __name__ == "__main__":
    seed_val = 1066
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    print(f"[INFO] Manual seed set to {seed_val}.")
    
#   Parse Arguments
    config = get_config("TRAIN_EEG-TEXT-BART")
    
    LOG_DIR = config["log_dir"]
    CKPT_DIR = config["ckpt_dir"]
    MODEL_NAME = config["model_name"]
    
    lr_init = config["initial_learning_rate"]
    lr_min = config["minimum_learning_rate"]
    lr_gamma = config["gamma_learning_rate"]
    lr_alt = config["alt_learning_rate"]
    
    device=config["device"]
    device_ids=config["device_ids"]
    
#   Load pretrained tokenizer
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    print("[INFO] Loaded tokenizer.")
    
#   Set up model
    learning_rate=lr_init
    model = INITIALIZE_MODEL(device=device, device_ids=device_ids).to(device)
    model = nn.DataParallel(model, device_ids=device_ids)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    print("[INFO] Initialized model.")
    print("[INFO] Model Components:")
    print(f"[INFO] {count_params(model)} TOTAL PARAMETERS.")
    print(f"[INFO] {count_params(model, trainable=True)} TRAINABLE PARAMETERS.")

#   Set up dataloaders
    dataloaders = INITIALIZE_DATALOADERS(
        keys=["ZuCo-BART"],
        bsz=[config["batch_size"]]
    )
    ZuCo_dataloader=dataloaders["ZuCo-BART"]
    print("[INFO] Intialized dataloaders.")
    
#   Set up log
    writer = SummaryWriter(log_dir=LOG_DIR)
    
#   Set up DSG
    dsg_tasks = DSGTasks()
    dsg_tasks.add_task(
        DSGTask(
            task_name="EEG-TEXT-BART",
            dataloader=ZuCo_dataloader,
            converge_lim=10,
            converge_threshold=0.001,
            div_threshold=0.1
            )
        )
    print("[INFO] Initialized DSG.")
    
#   Training Loop
    epoch_num = 0
    best_loss = 9e999
    print(f"|Epoch Num     |Task Name     |Current Loss            |Test Loss                 |Status            |Time                    |")
    while epoch_num < 50:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        for task in dsg_tasks.tasks:
            start_time = time.time()
            if task.name == "EEG-TEXT-BART":
                results = train_one_epoch(
                    model=model,
                    dataloader=task.dataloader,
                    optimizer=optimizer,
                    criterion=criterion,
                    tokenizer=tokenizer,
                    device=device,
                    device_ids=device_ids,
                    staging_device="cuda:1"
                    )
            model = results["model"]
            train_loss = results["train_loss"]
            dev_loss = results["dev_loss"]
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            if dev_loss < best_loss:
                best_loss = dev_loss
                torch.save(model.state_dict(), f"{CKPT_DIR}/{MODEL_NAME}_BEST.pt")
                

            if task.is_converged():
                print(f"|{epoch_num:12}|{task.name:12}|{train_loss:18.9f}|{dev_loss:18.9f}|CONVERGED     |{elapsed_time:12.2f} s|")
            elif task.is_diverged():
                print(f"|{epoch_num:12}|{task.name:12}|{train_loss:18.9f}|{dev_loss:18.9f}|DIVERGED        |{elapsed_time:12.2f} s|")
            else:
                print(f"|{epoch_num:12}|{task.name:12}|{train_loss:18.9f}|{dev_loss:18.9f}|TRAINING        |{elapsed_time:12.2f} s|")

            writer.add_scalar(f"{task.name} Train Loss", train_loss, epoch_num)
            writer.add_scalar(f"{task.name} Dev Loss", dev_loss, epoch_num)

            task.update(epoch_num, dev_loss)
        epoch_num += 1
        learning_rate *= lr_gamma
        if learning_rate < lr_min:
            learning_rate = lr_min
        
        if epoch_num % 5 == 0:
            torch.save(model.state_dict(), f"{CKPT_DIR}/{MODEL_NAME}_EPOCH{epoch_num}.pt")
    torch.save(model.state_dict(), f"{CKPT_DIR}/{MODEL_NAME}_FINAL.pt")
                
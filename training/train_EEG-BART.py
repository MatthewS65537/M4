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

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from transformers import BartTokenizer

def train_one_epoch(dataloader, model, optimizer, criterion, tokenizer, device="cuda"):
  results = {}
  for phase in ['train', 'dev']:
    if phase == 'train':
      model.train()  # Set model to training mode
    else:
      model.eval()   # Set model to evaluate mode

    running_loss = 0.0
    tot_cnt = 0

    # Iterate over data.
    current_data = dataloader[phase].load_data()
    while not current_data["reset"]:
      input_embeddings, seq_len, input_masks, input_mask_invert, target_ids, target_mask, sentiment_labels, sent_level_EEG = current_data["data"]
        
      input_embeddings_batch = input_embeddings.to(device).float()
      input_masks_batch = input_masks.to(device)
      input_mask_invert_batch = input_mask_invert.to(device)
      target_ids_batch = target_ids.to(device)

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
        args_dict=args_dict
        )

      loss = seq2seqLMoutput.loss # Use the BART language modeling loss
      
      # Backward + Optimize only if in training phase
      if phase == 'train':
          loss.backward()
          optimizer.step()

      # Compute stats
      running_loss += loss.item() * input_embeddings_batch.size()[0]
      tot_cnt += input_embeddings_batch.size()[0]
      current_data = dataloader[phase].load_data()

    epoch_loss = running_loss / tot_cnt

    results[f"{phase}_loss"] = epoch_loss
  results["model"] = model
  return results

if __name__ == "__main__":
  device="cuda"
  tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

  print("[INFO] Loaded tokenizer.")


  learning_rate = 5e-4
  model = INITIALIZE_MODEL(device=device).to(device)
  optimizer = optim.Adam(model.parameters(), lr=learning_rate)
  criterion = nn.CrossEntropyLoss()

  print("[INFO] Initialized model.")

  ZuCo_data = load_txt_data("./data/ZuCo")
  master_eeg, master_embeds = ZuCo_data["data"], ZuCo_data["targets"]
  del ZuCo_data
  bsz=16
  ZuCo_dataloader = {
    "train" : ZuCoDataloader(master_eeg["train"], master_embeds["train"], bsz=bsz, drop_last=True),
    "dev" : ZuCoDataloader(master_eeg["dev"], master_embeds["dev"], bsz=bsz, drop_last=True),
    "test" : ZuCoDataloader(master_eeg["test"], master_embeds["test"], bsz=bsz, drop_last=True)
  }
  del master_eeg, master_embeds
  print("[INFO] Prepared ZuCo dataloader.")

  log_dir = "./logs/EEG-TXT-BART"
  writer = SummaryWriter(log_dir=log_dir)

  dsg_tasks = DSGTasks()
  dsg_tasks.add_task(
    DSGTask(
      task_name="EEG-TEXT-BART",
      dataloader=ZuCo_dataloader,
      converge_lim=10,
      converge_threshold=0.0005,
      div_threshold=0.01
      )
    )
  print("[INFO] Initialized DSG.")

  epoch_num = 0
  print(f"|Epoch Num   |Task Name   |Current Loss      |Test Loss         |Status      |Time          |")
  while epoch_num < 10:
    for task in dsg_tasks.tasks:
      start_time = time.time()
      if task.name == "EEG-TEXT-BART":
        results = train_one_epoch(
          model=model,
          dataloader=task.dataloader,
          optimizer=optimizer,
          criterion=criterion,
          tokenizer=tokenizer,
          device=device
          )
      model = results["model"]
      train_loss = results["train_loss"]
      dev_loss = results["dev_loss"]
      end_time = time.time()
      elapsed_time = end_time - start_time

      if task.is_converged():
        print(f"|{epoch_num:12}|{task.name:12}|{train_loss:18.9f}|{dev_loss:18.9f}|CONVERGED   |{elapsed_time:12.6f} s|")
      elif task.is_diverged():
        print(f"|{epoch_num:12}|{task.name:12}|{train_loss:18.9f}|{dev_loss:18.9f}|DIVERGED    |{elapsed_time:12.6f} s|")
      else:
        print(f"|{epoch_num:12}|{task.name:12}|{train_loss:18.9f}|{dev_loss:18.9f}|TRAINING    |{elapsed_time:12.6f} s|")

      writer.add_scalar(f"{task.name} Train Loss", train_loss, epoch_num)
      writer.add_scalar(f"{task.name} Dev Loss", dev_loss, epoch_num)

      task.update(epoch_num, dev_loss)
    epoch_num += 1
import sys

sys.path.append("./models")
sys.path.append("./training")
sys.path.append("./utils")
sys.path.append("./ZuCo")

from eeg_encoder import *
from DSG import *
from load_data import *
from data import *
from dataloader import *

ZuCo_data = load_txt_data("./data/ZuCo")
master_eeg, master_embeds = ZuCo_data["data"], ZuCo_data["targets"]
del ZuCo_data
Brain2Image_data = load_img_data("./data/Brain2Image")
image_eeg_labels, img_net_dict = Brain2Image_data["data"], Brain2Image_data["targets"]
del Brain2Image_data

zuco_dataloader = ZuCoDataloader(master_eeg["dev"], master_embeds["dev"], bsz=64, drop_last=True)
# zuco_dataloader = ZuCoDataloader(master_eeg["train"], master_embeds["train"], bsz=64, drop_last=True)
image_net_dataloader = ImageNetDataloader(image_eeg_labels, img_net_dict, bsz=1, drop_last=True)

dsg_tasks = DSGTasks()
dsg_tasks.add_task(DSGTask("EEG-TXT", dataset=zuco_dataloader, converge_lim=10, div_threshold=0.01))
dsg_tasks.add_task(DSGTask("EEG-IMG", dataset=image_net_dataloader, converge_lim=10, div_threshold=0.01))

import torch
import torch.nn as nn
import torch.optim as optim

device = "cuda:0"

eeg_enc = EEGEncoder(txt_in_feat=840, img_in_feat=500, enc_feat=1024, dec_emb_sz=768, enc_nhead=8, enc_dim_ff=2048, num_enc_layers=8)
eeg_enc = nn.DataParallel(eeg_enc, device_ids=[0])

learning_rate = 5e-3
criterion = nn.CosineEmbeddingLoss()
optimizer = optim.Adam(eeg_enc.parameters(), lr=learning_rate)

epoch_num = 0
epoch_num = 0
while dsg_tasks.should_keep_training():
    epoch_num += 1
    for task in dsg_tasks.tasks:
        if task.should_keep_training():
            cur_loss = 0.0
            tot_cnt = 0
            if task.name == "EEG-TXT":
                zuco_data = zuco_dataloader.load_data()
                while not zuco_data["reset"]:
                    input_embeddings, seq_len, input_masks, input_mask_invert, target_ids, target_mask, sentiment_labels, sent_level_EEG = zuco_data["data"]
                    res = eeg_enc("TXT", input_embeddings.to(device).float(), input_masks.to(device), input_mask_invert.to(device))
                    embed = zuco_data["target"]

                    loss = criterion(res.to(device).float().view(embed.shape[0] * 77, 768), embed.to(device).float().view(embed.shape[0] * 77, 768), torch.ones(embed.shape[0] * 77).to(device))

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    cur_loss += loss.item()
                    tot_cnt += zuco_data["size"]
                    zuco_data = zuco_dataloader.load_data()
                print(epoch_num, "EEG-TXT")

            elif task.name == "EEG-IMG":
                # Needs Debugging
                image_net_data = image_net_dataloader.load_data()
                while not image_net_dataloader.reset():
                    input_data_batched = image_net_data["data"]
                    input_data_batched = input_data_batched[0]
                    target_batched = image_net_data["target"]

                    res = eeg_enc("IMG", input_data_batched.to(device).float(), pool_img_head=True)
                    loss = criterion(res.to(device).float().view(target_batched.shape[0], 768), target_batched.to(device).float().view(target_batched.shape[0], 768), torch.ones(target_batched.shape[0] * 77).to(device))

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    cur_loss += loss.item()
                    tot_cnt += image_net_data["size"]
                    image_net_data = image_net_dataloader.load_data()
                print(epoch_num, "EEG-IMG")
            else:
                print("[ERROR] BAD TASK NAME. CHECK NAMING OF TASKS AND TRAINING TO ENSURE ALL TASKS HAVE CORRESPONDING TRAINING IMPLEMENTED.")
                assert(False)

            cur_loss /= tot_cnt
            task.update(epoch_num, cur_loss)
            print(epoch_num, task.name, cur_loss)
import sys

sys.path.append("./models")
sys.path.append("./training")
sys.path.append("./testing")
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

zuco_dataloader = ZuCoDataloader(master_eeg["train"], master_embeds["train"], bsz=64, drop_last=True)
zuco_test_dataloader = ZuCoDataloader(master_eeg["test"], master_embeds["test"], bsz=64, drop_last=True)
image_net_dataloader = ImageNetDataloader(image_eeg_labels["train"], img_net_dict, bsz=1, drop_last=True)
image_net_test_dataloader = ImageNetDataloader(image_eeg_labels["test"], img_net_dict, bsz=1, drop_last=True)

dsg_tasks = DSGTasks()
dsg_tasks.add_task(DSGTask("EEG-TXT", dataset=zuco_dataloader, converge_lim=10, converge_threshold=0.005, div_threshold=0.01))
dsg_tasks.add_task(DSGTask("EEG-IMG", dataset=image_net_dataloader, converge_lim=10, converge_threshold=0.005, div_threshold=0.01))

from torch.utils.tensorboard import SummaryWriter

log_dir = "./logs/Train1"
writer = SummaryWriter(log_dir=log_dir)

import torch
import torch.nn as nn
import torch.optim as optim

device = "cuda"

eeg_enc = EEGEncoder(txt_in_feat=840, img_in_feat=128, enc_feat=1024, dec_emb_sz=768, enc_nhead=8, enc_dim_ff=2048, num_enc_layers=8).to(device)
# eeg_enc = nn.DataParallel(eeg_enc, device_ids=[0])
eeg_enc = nn.DataParallel(eeg_enc, device_ids=[0,1,2,3])

learning_rate = 1e-2
criterion = nn.CosineEmbeddingLoss()
optimizer = optim.Adam(eeg_enc.parameters(), lr=learning_rate)

import time
from test_Brain2Image import *
from test_ZuCo import *

epoch_num = 0
lr_alt = 0
set_final = False
dsg_tasks.reset_task()
dsg_tasks.set_convergence_threshold(learning_rate)
print(f"|Epoch Num   |Task Name   |Current Loss      |Test Loss         |Status      |Time          |")
while learning_rate > 1e-6 or set_final:
    while dsg_tasks.should_keep_training():
        epoch_num += 1
        for task in dsg_tasks.tasks:
            cur_loss = 0.0
            tot_cnt = 0
            test_loss = None

            start_time = time.time()
            if task.name == "EEG-TXT":
                zuco_data = zuco_dataloader.load_data()
                while not zuco_data["reset"]:
                    optimizer = optim.Adam(eeg_enc.parameters(), lr=learning_rate)
                    input_embeddings, seq_len, input_masks, input_mask_invert, target_ids, target_mask, sentiment_labels, sent_level_EEG = zuco_data["data"]
                    res = eeg_enc("TXT", input_embeddings.to(device).float(), input_masks.to(device), input_mask_invert.to(device))
                    embed = zuco_data["target"]

                    loss = criterion(res.to(device).float().view(embed.shape[0] * 77, 768), embed.to(device).float().view(embed.shape[0] * 77, 768), torch.ones(embed.shape[0] * 77).to(device))

                    optimizer.zero_grad()
                    if task.should_keep_training():
                      loss.backward()
                      optimizer.step()

                    cur_loss += loss.item() * zuco_data["size"]
                    tot_cnt += zuco_data["size"]
                    zuco_data = zuco_dataloader.load_data()
                test_loss = test_ZuCo(test_dataloader=zuco_test_dataloader, model=eeg_enc, loss_fn=criterion)["loss"]

            elif task.name == "EEG-IMG":
                optimizer = optim.Adam(eeg_enc.parameters(), lr=learning_rate * 1)
                image_net_data = image_net_dataloader.load_data()
                while not image_net_dataloader.reset():
                    input_data_batched = image_net_data["data"]
                    input_data_batched_converted = torch.zeros(tuple([len(input_data_batched)]) + input_data_batched[0].shape).to(device)
                    for i in range(len(input_data_batched)):
                      input_data_batched_converted[i] = input_data_batched[i].to(device)
                    target_batched = image_net_data["target"]
                    target_batched_converted = torch.zeros(tuple([len(target_batched)]) + target_batched[0].shape).to(device)
                    for i in range(len(target_batched)):
                      target_batched_converted[i] = target_batched[i].to(device)

                    res = eeg_enc("IMG", input_data_batched_converted.to(device).float(), pool_img_head=True)
                    loss = criterion(res.to(device).float().view(target_batched_converted.shape[0], 768),
                                      target_batched_converted.to(device).float().view(target_batched_converted.shape[0], 768),
                                      torch.ones(target_batched_converted.shape[0] * 77).to(device))

                    optimizer.zero_grad()

                    if task.should_keep_training():
                      loss.backward()
                      optimizer.step()

                    cur_loss += loss.item() * image_net_data["size"]
                    tot_cnt += image_net_data["size"]
                    image_net_data = image_net_dataloader.load_data()
                test_loss = test_Brain2Image(test_dataloader=image_net_test_dataloader, model=eeg_enc, loss_fn=criterion)["loss"]
            else:
                print("[ERROR] BAD TASK NAME. CHECK NAMING OF TASKS AND TRAINING TO ENSURE ALL TASKS HAVE CORRESPONDING TRAINING IMPLEMENTED.")
                assert(False)

            cur_loss /= tot_cnt
            # task.update(epoch_num, cur_loss)
            task.update(epoch_num, test_loss)
            writer.add_scalar(f"{task.name} Training Loss", cur_loss, epoch_num)
            writer.add_scalar(f"{task.name} Testing Loss", test_loss, epoch_num)

            end_time = time.time()
            elapsed_time = end_time - start_time
            if task.is_converged():
              print(f"|{epoch_num:12}|{task.name:12}|{cur_loss:18.9f}|{test_loss:18.9f}|CONVERGED   |{elapsed_time:12.6f} s|")
            elif task.is_diverged():
              print(f"|{epoch_num:12}|{task.name:12}|{cur_loss:18.9f}|{test_loss:18.9f}|DIVERGED    |{elapsed_time:12.6f} s|")
            else:
              print(f"|{epoch_num:12}|{task.name:12}|{cur_loss:18.9f}|{test_loss:18.9f}|TRAINING    |{elapsed_time:12.6f} s|")

    if learning_rate == 1e-6 and set_final:
        break
    if lr_alt % 2 == 0:
        learning_rate /= 4
    else:
        learning_rate *= 2
    if learning_rate < 1e-6:
        set_final = True
        learning_rate = 1e-6
    lr_alt += 1
    print(f"Convergence Acheived. Adjusted learning rate to {learning_rate}.")
    optimizer = optim.Adam(eeg_enc.parameters(), lr=learning_rate)
    dsg_tasks.reset_convergence()
    dsg_tasks.set_convergence_threshold(learning_rate)

torch.save(eeg_enc.state_dict(), "EEG_ENC.pt")
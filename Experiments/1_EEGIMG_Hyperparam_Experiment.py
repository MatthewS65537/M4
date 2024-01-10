print("[INFO] Starting 1_EEGIMG_Hyperparam_Experiment.py run.")
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

from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim

import time
from tqdm import tqdm
from test_Brain2Image import *
from test_ZuCo import *

import itertools

print("[INFO] Loaded libraries.")
print("[INFO] Loading datasets...")

Brain2Image_data = load_img_data("./data/Brain2Image")
image_eeg_labels, img_net_dict = Brain2Image_data["data"], Brain2Image_data["targets"]
del Brain2Image_data

print("[INFO] Loaded datasets.")

image_net_dataloader = ImageNetDataloader(image_eeg_labels["train"], img_net_dict, bsz=1, drop_last=True)
n_samples = len(image_eeg_labels["train"]["eeg"])
print(f"[INFO] Loaded {n_samples} samples for EEG-IMG train set.")
image_net_test_dataloader = ImageNetDataloader(image_eeg_labels["test"], img_net_dict, bsz=1, drop_last=True)
n_samples = len(image_eeg_labels["test"]["eeg"])
print(f"[INFO] Loaded {n_samples} samples for EEG-IMG test set.")

print("[INFO] Constructed dataloaders.")

device = "cuda"

list_converge_lim = [5]
list_initial_learning_rates = [5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
list_b_per_epoch = [64]

for converge_lim, initial_learning_rate, b_per_epoch in tqdm(list(itertools.product(list_converge_lim, list_initial_learning_rates, list_b_per_epoch))):
    eeg_enc = EEGEncoder(enc_feat=1024, dec_emb_sz=768, enc_nhead=8, enc_dim_ff=2048, num_enc_layers=8, device=device).to(device)
    eeg_enc.add_task("EEG-IMG", TaskHead(input_dim=128, output_dim=1024).to(device))
    eeg_enc = nn.DataParallel(eeg_enc, device_ids=[0,1,2,3])
    experiment_string = f"CLIM-{converge_lim}+ILR-{initial_learning_rate}+BPE{b_per_epoch}"
    
    print(f"Conducting Experiment {experiment_string}")
    
    log_dir = f"./Experiments/logs/1_EEGIMG_Hyperparam_Experiment_{experiment_string}"
    writer = SummaryWriter(log_dir=log_dir)
    dsg_tasks = DSGTasks()
    dsg_tasks.add_task(DSGTask("EEG-IMG", dataset=image_net_dataloader, converge_lim=converge_lim, converge_threshold=0.0005, div_threshold=0.01))

    learning_rate = initial_learning_rate
    min_lr = initial_learning_rate/16
    criterion = nn.CosineEmbeddingLoss()
    optimizer = optim.Adam(eeg_enc.parameters(), lr=learning_rate)

    print("[INFO] Built model.")

    print("[INFO] Start training.")

    epoch_num = 0
    lr_alt = 0
    num_batches_per_epoch = b_per_epoch
    set_final = False
    dsg_tasks.reset_task()
    image_net_dataloader.reset()
    print(f"|Epoch Num   |Task Name   |Current Loss      |Test Loss         |Status      |Time          |Convergence Rate|")
    while learning_rate > min_lr or set_final:
        while dsg_tasks.should_keep_training():
            epoch_num += 1
            for task in dsg_tasks.tasks:
                cur_loss = 0.0
                tot_cnt = 0
                test_loss = None

                start_time = time.time()

                for _ in range(num_batches_per_epoch):
                    image_net_data = image_net_dataloader.load_data()
                    input_data_batched = image_net_data["data"]
                    input_data_batched_converted = torch.zeros(tuple([len(input_data_batched)]) + input_data_batched[0].shape).to(device)
                    for i in range(len(input_data_batched)):
                        input_data_batched_converted[i] = input_data_batched[i].to(device)
                    target_batched = image_net_data["target"]
                    target_batched_converted = torch.zeros(tuple([len(target_batched)]) + target_batched[0].shape).to(device)
                    for i in range(len(target_batched)):
                        target_batched_converted[i] = target_batched[i].to(device)

                    res = eeg_enc(mode="EEG-IMG", input_data_batch=input_data_batched_converted.to(device).float(), pool_output=True)
                    loss = criterion(res.to(device).float().view(target_batched_converted.shape[0], 768),
                                      target_batched_converted.to(device).float().view(target_batched_converted.shape[0], 768),
                                      torch.ones(target_batched_converted.shape[0] * 77).to(device))

                    optimizer.zero_grad()

                    if task.should_keep_training():
                        loss.backward()
                        optimizer.step()

                    cur_loss += loss.item() * image_net_data["size"]
                    tot_cnt += image_net_data["size"]
                test_loss = test_Brain2Image(test_dataloader=image_net_test_dataloader, model=eeg_enc, loss_fn=criterion)["loss"]

                cur_loss /= tot_cnt
                task.update(epoch_num, test_loss)
                writer.add_scalar(f"{task.name} Training Loss", cur_loss, epoch_num)
                writer.add_scalar(f"{task.name} Testing Loss", test_loss, epoch_num)

                end_time = time.time()
                elapsed_time = end_time - start_time
                if task.is_converged():
                    print(f"|{epoch_num:12}|{task.name:12}|{cur_loss:18.9f}|{test_loss:18.9f}|CONVERGED   |{elapsed_time:12.6f} s|{task.convergence_rate:10.8f}/epoch|")
                elif task.is_diverged():
                    print(f"|{epoch_num:12}|{task.name:12}|{cur_loss:18.9f}|{test_loss:18.9f}|DIVERGED    |{elapsed_time:12.6f} s|{task.convergence_rate:10.8f}/epoch|")
                else:
                    print(f"|{epoch_num:12}|{task.name:12}|{cur_loss:18.9f}|{test_loss:18.9f}|TRAINING    |{elapsed_time:12.6f} s|{task.convergence_rate:8.6f}/epoch|")

        if set_final:
            break
        if lr_alt % 2 == 0:
            learning_rate /= 4
        else:
            learning_rate *= 2
        if learning_rate <= min_lr:
            set_final = True
            learning_rate = min_lr
        lr_alt += 1
        print(f"[TRAINING INFO] Convergence Acheived. Adjusted learning rate to {learning_rate}.")
        optimizer = optim.Adam(eeg_enc.parameters(), lr=learning_rate)
        dsg_tasks.reset_convergence()

    print("[INFO] Ended training.")

    torch.save(eeg_enc.state_dict(), f"./Checkpoints/1_EEGIMG_Hyperparam_Experiment/{experiment_string}.pt")

    print("[INFO] Saved checkpoint.")
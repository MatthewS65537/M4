import sys
sys.path.append("./models")
sys.path.append("./training")
sys.path.append("./testing")
sys.path.append("./utils")
sys.path.append("./ZuCo")
sys.path.append("./trainer")
import time

# Import Pytorch
import torch
import torch.nn as nn

import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# Import Pretrain Libraries (transformers + diffusers)
from transformers import BartTokenizer
from diffusers import AutoencoderKL

# Parallel Helper
from parallel import DataParallelModel, DataParallelCriterion

# Import Our Own Functions
from master_init import *
from DSG import *

from count_params import count_params

# Import Tasks
import EEG_TEXT_BART
# import EEG_TEXT_BART_SENTIMENT
# import EEG_IMG_DIFFUSION
# import EEG_IMG_CLASSIFICATION
# import PRETRAIN_EEG_IMG_CLIP_MATCHING
# import PRETRAIN_EEG_TEXT_CLIP_MATCHING
# import PRETRAIN_EEG_IMG_UNET
# import PRETRAIN_EEG_TEXT_UNET

print(f"[INFO] FETCHING CONFIGURATIONS.")
config = {
    "device" : "cuda",
    "device_ids" : [0,1,2,3],
    "staging_device" : "cuda",
    "num_epochs" : 50,
    "use_non_pytorch_parallel" : False,
    "test_run" : False,
    "live_evaluate" : True,
    "eval_interval" : 1,
}

device = config["device"]
device_ids = config["device_ids"]
staging_device = config["staging_device"]
num_epochs = config["num_epochs"]
use_non_pytorch_parallel = config["use_non_pytorch_parallel"]
test_run = config["test_run"]
live_evaluate = config["live_evaluate"]
eval_interval = config["eval_interval"]

print(f"[INFO] FINISHED CONFIGURATIONS.")

print(f"[INFO] INITIALIZING MODEL.")
copy_model = INITIALIZE_MODEL(device=None, device_ids=device_ids, dtype=torch.float32)
if use_non_pytorch_parallel:
    copy_model = DataParallelModel(copy_model, device_ids=device_ids).to(device)
else:
    copy_model = nn.DataParallel(copy_model, device_ids=device_ids).to(device)
state_dict = torch.load(f"./checkpoints/PretrainPlus/MMMM_70.pt")
copy_model.load_state_dict(state_dict, strict=False)

for name, param in copy_model.named_parameters():
    param.requires_grad=True
    if ("eeg_encoder" in name) or ("emb_unet" in name) or ("CLIP_text_encoder" in name):
        param.requires_grad=False
        continue
    if "branches" in name:
        if "EEG-TEXT-BART.body" in name:
            if not (("lora" in name) or ("encoder.layers.0" in name) or ("embed_positions" in name) or ("shared" in name)):
                param.requires_grad=False
                continue
        if "EEG-IMG-DIFFUSION.body" in name:
            if not ("lora" in name):
                param.requires_grad=False
                continue
    

dataset_dict = INITIALIZE_DATALOADERS(
    keys=["ZuCo-BART"],
    bsz=[256],
    dev_bsz=[256]
)
# if test_run:
#     for key, val in dataset_dict.items():
#         dataset_dict[key]["train"] = dataset_dict[key]["dev"] # Make things faster
#         dataset_dict[key]["dev"] = dataset_dict[key]["test"] # Mimic an unseen set

print(f"[INFO] LOADING PRETRAINS...")
BART_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")

dsg_tasks = DSGTasks()

dsg_tasks.add_task(
    DSGTask(
        task_name="EEG-TEXT-BART",
        dataset_tag="ZuCo-BART",
        criterion=nn.CrossEntropyLoss(), # Will use BART's own loss function (should also be CE Loss)
        optimizer=optim.Adam,
        learning_rate=5e-5,
        converge_lim=2,
        converge_threshold=0.005,
        div_threshold=0.005
        )
    )

import copy
import optuna
from tqdm import tqdm

best_loss = 3.407

def objective(trial):
    st_loss = -1
    global best_loss
    # Define the search space for learning rate
    lr = trial.suggest_float('lr', 1e-5, 1e-2,log=True)
    beta1 = trial.suggest_float('beta1', 0.6, 0.999)
    beta2 = trial.suggest_float('beta2', 0.7, 0.9999)
    eps = trial.suggest_float('eps', 1e-8, 1e-5,log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-8, 1e-5,log=True)
    gamma = trial.suggest_float('gamma', 0.9, 0.9999, log=True)
    gamma_up_ratio = trial.suggest_float('gamma_up_ratio', 0.1, 0.9)
    gamma_up = gamma_up_ratio * 1/(gamma ** 10)
    
    cur_best = 9e999
    best_epoch = 0


    model = copy.deepcopy(copy_model)
    
    for name, param in model.named_parameters():
        param.requires_grad=True
        if ("eeg_encoder" in name) or ("emb_unet" in name) or ("CLIP_text_encoder" in name):
            param.requires_grad=False
            continue
        if "branches" in name:
            if "EEG-TEXT-BART.body" in name:
                if not (("lora" in name) or ("encoder.layers.0" in name) or ("embed_positions" in name) or ("shared" in name)):
                    param.requires_grad=False
                    continue
            if "EEG-IMG-DIFFUSION.body" in name:
                if not ("lora" in name):
                    param.requires_grad=False
                    continue
                
    optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(beta1, beta2), eps=eps, weight_decay=weight_decay)
    scheduler1 = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    scheduler2 = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20,30,40], gamma=gamma_up)
    
    epoch = 0
    num_epochs = 50
    for epoch in tqdm(range(num_epochs)):
        for task in dsg_tasks.tasks:
#             start = time.time()
            if task.name == "EEG-TEXT-BART":
                args_dict = {
                    "model" : model,
                    "dataloader" : dataset_dict[task.dataset_tag],
                    "optimizer" : optimizer,
                    "tokenizer" : BART_tokenizer,
                    "criterion" : task.criterion, # Placeholder for BART LM Loss
                    "device" : device,
                    "device_ids" : device_ids,
                    "staging_device" : staging_device,
                }
                results = EEG_TEXT_BART.train(args_dict, using_non_pytorch_parallel=use_non_pytorch_parallel)
                model = results["model"]
                if results['dev_loss'] < cur_best:
                    cur_best = results['dev_loss']
                    best_epoch = epoch

                trial.report(results['dev_loss'], epoch)
                if st_loss == -1:
                    st_loss = results['dev_loss']
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
                # Early Stopping
                if results['dev_loss'] > st_loss * 1.1 or results['dev_loss'] > st_loss + 1 or best_epoch + 5 < epoch:
                    return results['dev_loss']
        scheduler1.step()
        scheduler2.step()
    
        if results['dev_loss'] < best_loss:
            torch.save(model.state_dict(), "./tune_checkpoints/BEST-BART.pt")
            best_loss = results['dev_loss']
        
    return results['dev_loss']

study = optuna.load_study(study_name="BART-50", storage="sqlite:///OPTUNA-DB/BART-50.db")
study.optimize(objective, n_trials=50)

pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

print("Study statistics: ")
print("  Number of finished trials: ", len(study.trials))
print("  Number of pruned trials: ", len(pruned_trials))
print("  Number of complete trials: ", len(complete_trials))

print("Best trial:")
trial = study.best_trial

print("  Value: ", trial.value)

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

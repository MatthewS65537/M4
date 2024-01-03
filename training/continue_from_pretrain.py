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
import EEG_TEXT_BART_SENTIMENT
import EEG_IMG_DIFFUSION
import EEG_IMG_CLASSIFICATION
import PRETRAIN_EEG_IMG_CLIP_MATCHING
import PRETRAIN_EEG_TEXT_CLIP_MATCHING

if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    
    # config = get_config() Implement later
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
        "log_dir" : "./logs"
    }
    
    device = config["device"]
    device_ids = config["device_ids"]
    staging_device = config["staging_device"]
    num_epochs = config["num_epochs"]
    use_non_pytorch_parallel = config["use_non_pytorch_parallel"]
    test_run = config["test_run"]
    live_evaluate = config["live_evaluate"]
    eval_interval = config["eval_interval"]
    log_dir = config["log_dir"]
    
    print(f"[INFO] FINISHED CONFIGURATIONS.")
    
    print(f"[INFO] INITIALIZING MODEL.")
    model = INITIALIZE_MODEL(device=None, device_ids=device_ids, dtype=torch.float32)
    if use_non_pytorch_parallel:
        model = DataParallelModel(model, device_ids=device_ids).to(device)
    else:
        model = nn.DataParallel(model, device_ids=device_ids).to(device)
    model.load_state_dict(torch.load(f"./checkpoints/Pretrains/MMMM_FINAL.pt"))
    dataset_dict = INITIALIZE_DATALOADERS(
        keys=["ZuCo-BART", "ZuCo-CLIP", "Brain2Image"],
        bsz=[256, 256, 1],
        dev_bsz=[256, 256, 1]
    )
    if test_run:
        for key, val in dataset_dict.items():
            dataset_dict[key]["train"] = dataset_dict[key]["dev"] # Make things faster
            
    train_writer = SummaryWriter(log_dir=f"{log_dir}/train-pretrain-continue")
    dev_writer = SummaryWriter(log_dir=f"{log_dir}/dev-pretraion-continue")
            
    print(f"[INFO] PARAMETER COUNT")
    print(f"[INFO] >>>> {count_params(model)} TOTAL PARAMETERS.")
    print(f"[INFO] >>>> {count_params(model,True)} TRAINABLE PARAMETERS.")
    print(f"[INFO] >>>> {count_params(model,False)} NON-TRAINABLE PARAMETERS.")
    
    # Load Pretrains
    print(f"[INFO] LOADING PRETRAINS...")
    BART_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").to(dtype=torch.float32)
    vae.requires_grad_(False)
    print(f"[INFO] LOADED PRETRAINS.")
    
#     latent_dict = {}
    # Decision not to put dataloders into DSGTask() object
    # Will take too much space and some datasets repeatedly used for
    # different tasks.
    print(f"[INFO] SETTING UP TRAINING TASKS...")
    dsg_tasks = DSGTasks()
    
    dsg_tasks.add_task(
    DSGTask(
        task_name="PRETRAIN-EEG-TEXT-CLIP-MATCHING",
        dataset_tag="ZuCo-CLIP",
        criterion=nn.KLDivLoss(reduction="batchmean"), # Symmetrized with Lambda inside train()
        optimizer=optim.Adam,
        learning_rate=2e-5,
        converge_lim=2,
        converge_threshold=0.0005,
        div_threshold=0.02
        )
    )

    dsg_tasks.add_task(
    DSGTask(
        task_name="PRETRAIN-EEG-IMG-CLIP-MATCHING",
        dataset_tag="Brain2Image",
        criterion=nn.KLDivLoss(reduction="batchmean"), # Symmetrized with Lambda inside train()
        optimizer=optim.Adam,
        learning_rate=2.5e-5,
        converge_lim=2,
        converge_threshold=0.005,
        div_threshold=0.005
        )
    )
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

    dsg_tasks.add_task(
        DSGTask(
            task_name="EEG-IMG-DIFFUSION",
            dataset_tag="Brain2Image",
            criterion=nn.MSELoss(), # For Noise predicted by latents
            optimizer=optim.Adam,
            learning_rate=5e-5,
            converge_lim=2,
            converge_threshold=0.005,
            div_threshold=0.005
            )
        )

    dsg_tasks.add_task(
        DSGTask(
            task_name="EEG-IMG-CLASSIFICATION",
            dataset_tag="Brain2Image",
            criterion=nn.CrossEntropyLoss(), # CE Loss for 40 classes
            optimizer=optim.Adam,
            learning_rate=5e-5,
            converge_lim=2,
            converge_threshold=0.005,
            div_threshold=0.005
            )
        )

    dsg_tasks.add_task(
        DSGTask(
            task_name="EEG-TEXT-BART-SENTIMENT",
            dataset_tag="ZuCo-BART",
            criterion=nn.CrossEntropyLoss(), # CE Loss for Tenary Sentiment
            optimizer=optim.Adam,
            learning_rate=5e-5,
            converge_lim=2,
            converge_threshold=0.005,
            div_threshold=0.005
            )
        )
    
    print(f"[INFO] FINISHED SETTING UP TRAINING TASKS.")
    
    lr_scale = 1.0
    lr_alt = 0
    gamma = 0.5
    epoch = 0
    
    print(f"[INFO] STARTING TRAINING.")
    while epoch < num_epochs:
        print(f"Epoch {epoch}")
        start = time.time()
        for task in dsg_tasks.tasks:
            if task.name == "PRETRAIN-EEG-TEXT-CLIP-MATCHING":
                args_dict = {
                    "model" : model,
                    "dataloader" : dataset_dict[task.dataset_tag],
                    "optimizer" : task.optimizer(model.parameters(), lr=task.learning_rate * lr_scale),
                    "tokenizer" : BART_tokenizer,
                    "criterion" : task.criterion,
                    "device" : device,
                    "device_ids" : device_ids,
                    "staging_device" : staging_device,
                    "dev_bsz" : 256,
                    "bool_eval" : True,
                    "temperature" : 20
                }
                results = PRETRAIN_EEG_TEXT_CLIP_MATCHING.train(args_dict, using_non_pytorch_parallel=use_non_pytorch_parallel)
                if epoch % eval_interval == 0 and live_evaluate:
                    print(f">>>>>>>> TRAIN ACCURACY: {results['train_accuracy'] * 100 : 8.4f} % DEV ACCURACY: {results['dev_accuracy'] * 100 : 8.4f} %")
                    train_writer.add_scalar(f"{task.name} Accuracy", results['train_accuracy'] * 100, epoch)
                    dev_writer.add_scalar(f"{task.name} Accuracy", results['dev_accuracy'] * 100, epoch)
                    

            elif task.name == "PRETRAIN-EEG-IMG-CLIP-MATCHING":
                args_dict = {
                    "model" : model,
                    "dataloader" : dataset_dict[task.dataset_tag],
                    "optimizer" : task.optimizer(model.parameters(), lr=task.learning_rate * lr_scale),
                    "criterion" : task.criterion,
                    "device" : device,
                    "device_ids" : device_ids,
                    "staging_device" : staging_device,
                    "bsz" : 256,
                    "bool_eval" : True,
                    "temperature" : 20
                }
                results = PRETRAIN_EEG_IMG_CLIP_MATCHING.train(args_dict, using_non_pytorch_parallel=use_non_pytorch_parallel)
                if epoch % eval_interval == 0 and live_evaluate:
                    print(f">>>>>>>> TRAIN ACCURACY: {results['train_accuracy'] * 100 : 8.4f} % DEV ACCURACY: {results['dev_accuracy'] * 100 : 8.4f} %")
                    train_writer.add_scalar(f"{task.name} Accuracy", results['train_accuracy'] * 100, epoch)
                    dev_writer.add_scalar(f"{task.name} Accuracy", results['dev_accuracy'] * 100, epoch)
                
                
            elif task.name == "EEG-TEXT-BART":
                args_dict = {
                    "model" : model,
                    "dataloader" : dataset_dict[task.dataset_tag],
                    "optimizer" : task.optimizer(model.parameters(), lr=task.learning_rate * lr_scale),
                    "tokenizer" : BART_tokenizer,
                    "criterion" : task.criterion, # Placeholder for BART LM Loss
                    "device" : device,
                    "device_ids" : device_ids,
                    "staging_device" : staging_device,
                }
                results = EEG_TEXT_BART.train(args_dict, using_non_pytorch_parallel=use_non_pytorch_parallel)

            elif task.name == "EEG-TEXT-BART-SENTIMENT":
                args_dict = {
                    "model" : model,
                    "dataloader" : dataset_dict[task.dataset_tag],
                    "optimizer" : task.optimizer(model.parameters(), lr=task.learning_rate * lr_scale),
                    "criterion" : task.criterion,
                    "tokenizer" : BART_tokenizer,
                    "device" : device,
                    "device_ids" : device_ids,
                    "staging_device" : staging_device,
                    "temperature" : 0.04
                }
                results = EEG_TEXT_BART_SENTIMENT.train(args_dict, using_non_pytorch_parallel=use_non_pytorch_parallel)
                
            elif task.name == "EEG-IMG-DIFFUSION":
                args_dict = {
                    "model" : model,
                    "dataloader" : dataset_dict[task.dataset_tag],
                    "optimizer" : task.optimizer(model.parameters(), lr=task.learning_rate * lr_scale),
                    "criterion" : task.criterion,
                    "device" : device,
                    "device_ids" : device_ids,
                    "staging_device" : staging_device,
                    "vae" : vae,
                    "bsz" : 64
#                     "latent_dict" : latent_dict
                }
                results = EEG_IMG_DIFFUSION.train(args_dict, using_non_pytorch_parallel=use_non_pytorch_parallel)
#                 latent_dict = results["latent_dict"]
                dev_writer.add_scalar(f"{task.name} Loss", results['dev_loss'], epoch)
            elif task.name == "EEG-IMG-CLASSIFICATION":
                args_dict = {
                    "model" : model,
                    "dataloader" : dataset_dict[task.dataset_tag],
                    "optimizer" : task.optimizer(model.parameters(), lr=task.learning_rate),
                    "criterion" : task.criterion,
                    "device" : device,
                    "device_ids" : device_ids,
                    "staging_device" : staging_device,
                    "temperature" : 0.04,
                    "bsz" : 256
                }
                results = EEG_IMG_CLASSIFICATION.train(args_dict, using_non_pytorch_parallel=use_non_pytorch_parallel)
            else:
                print(f"[WARNING] Task {task.name} not found. Skipping.")
                continue
                
            print(f">>>> {task.name} | TRAIN: {results['train_loss']} DEV: {results['dev_loss']} TIME: {time.time() - start:.2f} SECONDS")
            train_writer.add_scalar(f"{task.name} Loss", results['train_loss'], epoch)
            dev_writer.add_scalar(f"{task.name} Loss", results['dev_loss'], epoch)
            task.update(epoch, results['dev_loss'])
            if task.should_keep_training():
                model = results["model"]
                
        if not DSG_tasks.should_keep_training():
            DSG_tasks.reset_convergence()
            if lr_alt % 2 == 0:
                lr_scale *= gamma ** 2
            else:
                lr_scale /= gamma
            lr_alt += 1
            print("CONVERGENCE ACHEIVED ON ALL TASKS")
            
        epoch += 1
        if epoch % 10 == 0:
            torch.save(model.state_dict(), f"./checkpoints/DSG/MMMM_{epoch}.pt")
    torch.save(model.state_dict(), f"./checkpoints/DSG/MMMM_FINAL.pt")
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
import PRETRAIN_EEG_IMG_UNET
import PRETRAIN_EEG_TEXT_UNET

if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    
    # config = get_config() Implement later
    print(f"[INFO] FETCHING CONFIGURATIONS.")
    config = {
        "device" : "cuda",
        "device_ids" : [0,1,2,3],
        "staging_device" : "cuda",
        "num_epochs" : 100,
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
    
    dataset_dict = INITIALIZE_DATALOADERS(
        keys=["ZuCo-BART", "ZuCo-CLIP", "Brain2Image"],
        bsz=[256, 256, 1],
        dev_bsz=[256, 256, 1]
    )
    if test_run:
        for key, val in dataset_dict.items():
            dataset_dict[key]["train"] = dataset_dict[key]["dev"] # Make things faster
            dataset_dict[key]["dev"] = dataset_dict[key]["test"] # Mimic an unseen set
            
    train_writer = SummaryWriter(log_dir=f"{log_dir}/train-PRETRAIN-TUNED-FINAL2")
    dev_writer = SummaryWriter(log_dir=f"{log_dir}/dev-PRETRAIN-TUNED-FINAL2")
            
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
    
    # Decision not to put dataloders into DSGTask() object
    # Will take too much space and some datasets repeatedly used for
    # different tasks.
    print(f"[INFO] SETTING UP TRAINING TASKS...")
    dsg_tasks = DSGTasks()
    
#     hparams = {'lr': 1.4201593837329537e-05, 'beta1': 0.7014348222942296, 'beta2': 0.9743954103580398, 'eps': 6.45133637419436e-07, 'weight_decay': 7.290973770285979e-06, 'gamma': 0.8167180662558771}
    
    # hparams are tuned
    dsg_tasks.add_task(
        DSGTask(
            task_name="PRETRAIN-EEG-TEXT-CLIP-MATCHING",
            dataset_tag="ZuCo-CLIP",
            criterion=nn.KLDivLoss(reduction="batchmean"), # Symmetrized with Lambda inside train()
            optimizer=optim.AdamW,
            learning_rate=1.5e-5,
            converge_lim=2,
            converge_threshold=0.005,
            div_threshold=0.01
            )
        )
    
    dsg_tasks.add_task(
        DSGTask(
            task_name="PRETRAIN-EEG-IMG-CLIP-MATCHING",
            dataset_tag="Brain2Image",
            criterion=nn.KLDivLoss(reduction="batchmean"), # Symmetrized with Lambda inside train()
            optimizer=optim.AdamW,
            learning_rate=1e-5,
            converge_lim=2,
            converge_threshold=0.005,
            div_threshold=0.01
            )
        )
    
#     dsg_tasks.add_task(
#         DSGTask(
#             task_name="PRETRAIN-EEG-TEXT-UNET",
#             dataset_tag="ZuCo-CLIP",
#             criterion=nn.CosineEmbeddingLoss(),
#             optimizer=optim.AdamW,
#             learning_rate=5e-5,
#             converge_lim=2,
#             converge_threshold=0.005,
#             div_threshold=0.01
#             )
#         )
    
#     dsg_tasks.add_task(
#         DSGTask(
#             task_name="PRETRAIN-EEG-IMG-UNET",
#             dataset_tag="Brain2Image",
#             criterion=nn.CosineEmbeddingLoss(),
#             optimizer=optim.AdamW,
#             learning_rate=3e-5,
#             converge_lim=2,
#             converge_threshold=0.005,
#             div_threshold=0.01
#             )
#         )
    
    print(f"[INFO] FINISHED SETTING UP TRAINING TASKS.")
    
    lr_scale = 1.0
    epoch = 0
    gamma = 0.935
    num_epochs = 100
    print(f"[INFO] STARTING TRAINING.")
    while epoch < num_epochs:
        print(f"Epoch {epoch}")
        if epoch < 10:
            lr_scale = 0.2 + epoch * 0.08
        else:
            lr_scale = max(gamma ** (epoch - 10), 1e-3)
            
            
        
        epc_start = time.time()
        for task in dsg_tasks.tasks:
            start = time.time()
            if task.name == "PRETRAIN-EEG-TEXT-CLIP-MATCHING":
                for name, param in model.named_parameters():
                    if "eeg_encoder" in name:
                        param.requires_grad = True
                    if "emb_unet" in name:
                        param.requires_grad = False
                args_dict = {
                    "model" : model,
                    "dataloader" : dataset_dict[task.dataset_tag],
                    "optimizer" : task.optimizer(model.parameters(), lr=task.learning_rate * lr_scale, betas=(0.7, 0.975), eps=5e-7, weight_decay=7.5e-6),
                    "tokenizer" : BART_tokenizer,
                    "criterion" : task.criterion,
                    "device" : device,
                    "device_ids" : device_ids,
                    "staging_device" : staging_device,
                    "use_unet" : False,
                    "dev_bsz" : 256,
                    "bool_eval" : True,
                    "temperature" : 25,
                }
                results = PRETRAIN_EEG_TEXT_CLIP_MATCHING.train(args_dict, using_non_pytorch_parallel=use_non_pytorch_parallel)
                model = results["model"]
                if "train_accuracy" in results:
                    if epoch % eval_interval == 0 and live_evaluate:
                        print(f">>>>>>>> TRAIN ACCURACY: {results['train_accuracy'] * 100 : 8.4f} % DEV ACCURACY: {results['dev_accuracy'] * 100 : 8.4f} %")
                        train_writer.add_scalar(f"{task.name} Accuracy", results['train_accuracy'] * 100, epoch)
                        dev_writer.add_scalar(f"{task.name} Accuracy", results['dev_accuracy'] * 100, epoch)

                print(f">>>> {task.name} | TRAIN: {results['train_loss']} DEV: {results['dev_loss']} TIME: {time.time() - start:.2f} SECONDS")
                train_writer.add_scalar(f"{task.name} Loss", results['train_loss'], epoch)
                dev_writer.add_scalar(f"{task.name} Loss", results['dev_loss'], epoch)
                
            elif task.name == "PRETRAIN-EEG-IMG-CLIP-MATCHING":
                for name, param in model.named_parameters():
                    if "eeg_encoder" in name:
                        param.requires_grad = True
                    if "emb_unet" in name:
                        param.requires_grad = False
                args_dict = {
                    "model" : model,
                    "dataloader" : dataset_dict[task.dataset_tag],
                    "optimizer" : task.optimizer(model.parameters(), lr=task.learning_rate * lr_scale, betas=(0.7, 0.975), eps=5e-7, weight_decay=7.5e-6),
                    "criterion" : task.criterion,
                    "device" : device,
                    "device_ids" : device_ids,
                    "staging_device" : staging_device,
                    "use_unet" : False,
                    "bsz" : 256,
                    "bool_eval" : True,
                    "temperature" : 20,
                }
                results = PRETRAIN_EEG_IMG_CLIP_MATCHING.train(args_dict, using_non_pytorch_parallel=use_non_pytorch_parallel)
                model = results["model"]
                if "train_accuracy" in results:
                    if epoch % eval_interval == 0 and live_evaluate:
                        print(f">>>>>>>> TRAIN ACCURACY: {results['train_accuracy'] * 100 : 8.4f} % DEV ACCURACY: {results['dev_accuracy'] * 100 : 8.4f} %")
                        train_writer.add_scalar(f"{task.name} Accuracy", results['train_accuracy'] * 100, epoch)
                        dev_writer.add_scalar(f"{task.name} Accuracy", results['dev_accuracy'] * 100, epoch)

                print(f">>>> {task.name} | TRAIN: {results['train_loss']} DEV: {results['dev_loss']} TIME: {time.time() - start:.2f} SECONDS")
                train_writer.add_scalar(f"{task.name} Loss", results['train_loss'], epoch)
                dev_writer.add_scalar(f"{task.name} Loss", results['dev_loss'], epoch)
                model = results["model"]  
                
#             elif task.name == "PRETRAIN-EEG-TEXT-UNET":
#                 if epoch < 50:
#                     continue
#                 elif epoch < 75:
#                     for name, param in model.named_parameters():
#                         if "eeg_encoder" in name:
#                             param.requires_grad = False
#                         if "emb_unet" in name:
#                             param.requires_grad = True
#                 args_dict = {
#                     "model" : model,
#                     "dataloader" : dataset_dict[task.dataset_tag],
#                     "optimizer" : task.optimizer(model.parameters(), lr=task.learning_rate * lr_scale),
#                     "tokenizer" : BART_tokenizer,
#                     "criterion" : task.criterion,
#                     "device" : device,
#                     "device_ids" : device_ids,
#                     "staging_device" : staging_device,
#                     "bsz" : 256,
#                 }
#                 results = PRETRAIN_EEG_TEXT_UNET.train(args_dict, using_non_pytorch_parallel=use_non_pytorch_parallel)
#                 model = results["model"]
#                 if "train_accuracy" in results:
#                     if epoch % eval_interval == 0 and live_evaluate:
#                         print(f">>>>>>>> TRAIN ACCURACY: {results['train_accuracy'] * 100 : 8.4f} % DEV ACCURACY: {results['dev_accuracy'] * 100 : 8.4f} %")
#                         train_writer.add_scalar(f"{task.name} Accuracy", results['train_accuracy'] * 100, epoch)
#                         dev_writer.add_scalar(f"{task.name} Accuracy", results['dev_accuracy'] * 100, epoch)

#                 print(f">>>> {task.name} | TRAIN: {results['train_loss']} DEV: {results['dev_loss']} TIME: {time.time() - start:.2f} SECONDS")
#                 train_writer.add_scalar(f"{task.name} Loss", results['train_loss'], epoch)
#                 dev_writer.add_scalar(f"{task.name} Loss", results['dev_loss'], epoch)
#             elif task.name == "PRETRAIN-EEG-IMG-UNET":
#                 if epoch < 50:
#                     continue
#                 elif epoch < 75:
#                     for name, param in model.named_parameters():
#                         if "eeg_encoder" in name:
#                             param.requires_grad = False
#                         if "emb_unet" in name:
#                             param.requires_grad = True
#                 args_dict = {
#                     "model" : model,
#                     "dataloader" : dataset_dict[task.dataset_tag],
#                     "optimizer" : task.optimizer(model.parameters(), lr=task.learning_rate * lr_scale),
#                     "criterion" : task.criterion,
#                     "device" : device,
#                     "device_ids" : device_ids,
#                     "staging_device" : staging_device,
#                     "bsz" : 256,
#                 }
#                 results = PRETRAIN_EEG_IMG_UNET.train(args_dict, using_non_pytorch_parallel=use_non_pytorch_parallel)
#                 model = results["model"]
#                 if "train_accuracy" in results:
#                     if epoch % eval_interval == 0 and live_evaluate:
#                         print(f">>>>>>>> TRAIN ACCURACY: {results['train_accuracy'] * 100 : 8.4f} % DEV ACCURACY: {results['dev_accuracy'] * 100 : 8.4f} %")
#                         train_writer.add_scalar(f"{task.name} Accuracy", results['train_accuracy'] * 100, epoch)
#                         dev_writer.add_scalar(f"{task.name} Accuracy", results['dev_accuracy'] * 100, epoch)

#                 print(f">>>> {task.name} | TRAIN: {results['train_loss']} DEV: {results['dev_loss']} TIME: {time.time() - start:.2f} SECONDS")
#                 train_writer.add_scalar(f"{task.name} Loss", results['train_loss'], epoch)
#                 dev_writer.add_scalar(f"{task.name} Loss", results['dev_loss'], epoch)
#                 model = results["model"]                
            else:
                print(f"[WARNING] Task {task.name} not found. Skipping.")
                continue
                
#             del results
#             torch.cuda.empty_cache()
            
        print(f"TOT TIME: {time.time() - epc_start:.2f} SECONDS")
        if (epoch + 1) % 5 == 0 or epoch == 0:
            torch.save(model.state_dict(), f"./checkpoints/Pretrain_pe/MMMM_{epoch}.pt")
            print('*** checkpoint saved ***')
        epoch += 1
        
    torch.save(model.state_dict(), f"./checkpoints/Pretrain_pe/MMMM_FINAL.pt") 
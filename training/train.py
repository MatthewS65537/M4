import sys
sys.path.append("./models")
sys.path.append("./training")
sys.path.append("./testing")
sys.path.append("./utils")
sys.path.append("./ZuCo")
sys.path.append("./trainer")

# Import Pytorch
import torch
import torch.nn as nn
import torch.optim as optim

# Import Pretrain Libraries (transformers + diffusers)
from transformers import BartTokenizer
from diffusers import AutoencoderKL

# Import Our Own Functions
from master_init import *
from DSG import *

from count_params import count_params

# Import Tasks
import EEG_TEXT_BART
import EEG_TEXT_BART_SENTIMENT
import EEG_IMG_DIFFUSION
import EEG_IMG_CLASSIFICATION

if __name__ == "__main__":
#     torch.set_default_dtype(torch.float16)
    
    # config = get_config() Implement later
    print(f"[INFO] FETCHING CONFIGURATIONS.")
    config = {
        "device" : "cuda",
        "device_ids" : [0,1,2,3],
        "staging_device" : "cuda:0",
        "num_epochs" : 100
    }
    
    device = config["device"]
    device_ids = config["device_ids"]
    staging_device = config["staging_device"]
    num_epochs = config["num_epochs"]
    
    print(f"[INFO] FINISHED CONFIGURATIONS.")
    
    print(f"[INFO] INITIALIZING MODEL.")
    model = INITIALIZE_MODEL(device=device, device_ids=device_ids)
    model = nn.DataParallel(model, device_ids=device_ids)
    dataset_dict = INITIALIZE_DATALOADERS(
        keys=["ZuCo-BART", "Brain2Image"],
        bsz=[512, 1]
    )
    print(f"[INFO] PARAMETER COUNT")
    print(f"[INFO] >>>> {count_params(model)} TOTAL PARAMETERS.")
    print(f"[INFO] >>>> {count_params(model,True)} TRAINABLE PARAMETERS.")
    print(f"[INFO] >>>> {count_params(model,False)} NON-TRAINABLE PARAMETERS.")
    
    # Load Pretrains
    print(f"[INFO] LOADING PRETRAINS...")
    BART_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").to(device, dtype=torch.float16)
    vae.requires_grad_(False)
    print(f"[INFO] LOADED PRETRAINS.")
    
    # Decision not to put dataloders into DSGTask() object
    # Will take too much space and some datasets repeatedly used for
    # different tasks.
    print(f"[INFO] SETTING UP TRAINING TASKS...")
    dsg_tasks = DSGTasks()
    dsg_tasks.add_task(
        DSGTask(
            task_name="EEG-TEXT-BART",
            dataset_tag="ZuCo-BART",
            criterion=nn.CrossEntropyLoss(), # Will use BART's own loss function (should also be CE Loss)
            optimizer=optim.Adam,
            learning_rate=5e-3,
            converge_lim=2,
            converge_threshold=0.05,
            div_threshold=0.01
            )
        )

    dsg_tasks.add_task(
        DSGTask(
            task_name="EEG-IMG-DIFFUSION",
            dataset_tag="Brain2Image",
            criterion=nn.MSELoss(), # For Noise predicted by latents
            optimizer=optim.Adam,
            learning_rate=5e-3,
            converge_lim=2,
            converge_threshold=0.005,
            div_threshold=0.01
            )
        )

    dsg_tasks.add_task(
        DSGTask(
            task_name="EEG-IMG-CLASSIFICATION",
            dataset_tag="Brain2Image",
            criterion=nn.CrossEntropyLoss(), # CE Loss for 40 classes
            optimizer=optim.Adam,
            learning_rate=5e-3,
            converge_lim=2,
            converge_threshold=0.005,
            div_threshold=0.01
            )
        )

    dsg_tasks.add_task(
        DSGTask(
            task_name="EEG-TEXT-BART-SENTIMENT",
            dataset_tag="ZuCo-BART",
            criterion=nn.CrossEntropyLoss(), # CE Loss for Tenary Sentiment
            optimizer=optim.Adam,
            learning_rate=5e-3,
            converge_lim=2,
            converge_threshold=0.005,
            div_threshold=0.01
            )
        )
    print(f"[INFO] FINISHED SETTING UP TRAINING TASKS.")
    
    print(f"[INFO] STARTING TRAINING.")
    for epoch_num in range(num_epochs):
        print(f"Epoch {epoch_num}")
        for task in dsg_tasks.tasks:
            if task.name == "EEG-TEXT-BART":
                args_dict = {
                    "model" : model,
                    "dataloader" : dataset_dict[task.dataset_tag],
                    "optimizer" : optim.Adam(model.parameters(), lr=task.learning_rate),
                    "tokenizer" : BART_tokenizer,
                    "criterion" : task.criterion,
                    "device" : device,
                    "device_ids" : device_ids,
                    "staging_device" : staging_device
                }
                results = EEG_TEXT_BART.train(args_dict)
                model = results["model"]
                print(f">>>>{task.name} TRAIN: {results['train_loss']} DEV: {results['dev_loss']}")
            elif task.name == "EEG-TEXT-BART-SENTIMENT":
                args_dict = {
                    "model" : model,
                    "dataloader" : dataset_dict[task.dataset_tag],
                    "optimizer" : task.optimizer(model.parameters(), lr=task.learning_rate),
                    "criterion" : task.criterion,
                    "tokenizer" : BART_tokenizer,
                    "device" : device,
                    "device_ids" : device_ids,
                    "staging_device" : staging_device
                }
                results = EEG_TEXT_BART_SENTIMENT.train(args_dict)
                model = results["model"]
                print(f">>>>{task.name} TRAIN: {results['train_loss']} DEV: {results['dev_loss']}")
            elif task.name == "EEG-IMG-DIFFUSION":
                args_dict = {
                    "model" : model,
                    "dataloader" : dataset_dict[task.dataset_tag],
                    "optimizer" : task.optimizer(model.parameters(), lr=task.learning_rate),
                    "criterion" : task.criterion,
                    "device" : device,
                    "device_ids" : device_ids,
                    "staging_device" : staging_device,
                    "vae" : vae
                }
                results = EEG_IMG_DIFFUSION.train(args_dict)
                model = results["model"]
                print(f">>>>{task.name} TRAIN: {results['train_loss']} DEV: {results['dev_loss']}")
            elif task.name == "EEG-IMG-BRAIN2IMAGE-CLASSIFICATION":
                args_dict = {
                    "model" : model,
                    "dataloader" : dataset_dict[task.dataset_tag],
                    "optimizer" : task.optimizer(model.parameters(), lr=task.learning_rate),
                    "criterion" : task.criterion,
                    "device" : device,
                    "device_ids" : device_ids,
                    "staging_device" : staging_device
                }
                results = EEG_IMG_BRAIN2IMAGE_CLASSIFICATION.train(args_dict)
                model = results["model"]
                print(f">>>>{task.name} TRAIN: {results['train_loss']} DEV: {results['dev_loss']}")
            else:
                print("[WARNING] Task {task.name} not found. Skipping.")
            
            if epoch_num % 10 == 0:
                torch.save(model.state_dict(), "./BUF.pt")
                with open("./log.txt", "w") as f:
                    f.write(f"Epoch No. {epoch_num}\n")
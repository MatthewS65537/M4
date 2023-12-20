import sys
sys.path.append("./models")
sys.path.append("./training")
sys.path.append("./testing")
sys.path.append("./utils")
sys.path.append("./ZuCo")
sys.path.append("./trainer")

import torch
import torch.nn as nn
import torch.optim as optim

from master_init import *
from DSG import *

import EEG_TEXT_BART

def master_train(mode, args_dict):
    if mode == "EEG-TEXT-BART":
        result = EEG_TEXT_BART.train(args_dict)

if __name__ == "__main__":
    # config = get_config() Implement later
    config = {
        "device" : "cuda:0",
        "device_ids" : [0]
    }
    device = config["device"]
    device_ids = config["device_ids"]

    model = INITIALIZE_MODEL(device=device, device_ids=device_ids).to(device)
    dataset_dict = INITIALIZE_DATALOADERS(
        keys=["Zuco-BART", "Brain2Image"],
        bsz=[512, 1]
    )

    # Decision not to put dataloders into DSGTask() object
    # Will take too much space and some datasets repeatedly used for
    # different tasks.
    dsg_tasks = DSGTasks()
    dsg_tasks.add_task(
        DSGTask(
            task_name="EEG-TXT-BART",
            dataset_tag="ZuCo-BART",
            criterion=nn.CrossEntropyLoss(), # Will use BART's own loss function (should also be CE Loss)
            optimizer=optim.Adam,
            converge_lim=2,
            converge_threshold=0.05,
            div_threshold=0.01
            )
        )

    dsg_tasks.add_task(
        DSGTask(
            task_name="EEG-IMG-DIFFUSION",
            dataset_tag="Brain2Image",
            criterion=None, # Determine Later
            optimizer=optim.Adam,
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
            converge_lim=2,
            converge_threshold=0.005,
            div_threshold=0.01
            )
        )

    dsg_tasks.add_task(
        DSGTask(
            task_name="EEG-TXT-BART-SENTIMENT",
            dataset_tag="ZuCo-BART",
            criterion=nn.CrossEntropyLoss(), # CE Loss for Tenary Sentiment
            optimizer=optim.Adam,
            converge_lim=2,
            converge_threshold=0.005,
            div_threshold=0.01
            )
        )


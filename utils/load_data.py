import pickle
import torch

def load_txt_data(dir="./data"):
    with open(f"{dir}/ZuCoProcessedDatasetDict.pkl", "rb") as f:
        master_eeg = pickle.load(f)
    with open(f"{dir}/ZuCoTargetStringsEmbeds.pkl", "rb") as f:
        master_embeds = pickle.load(f)
    return {"data" : master_eeg, "targets" : master_embeds}

def load_img_data(dir="./data"):
    with open(f"{dir}/imageNet_labeled_eeg.pkl", "rb") as f:
        image_eeg_labels = pickle.load(f)
    with open(f"{dir}/image_net_dict.pkl", "rb") as f:
        img_net_dict = pickle.load(f)
    return {"data" : labeled_eeg, "targets" : image_net_dict}
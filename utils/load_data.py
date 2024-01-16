import pickle
import torch

def load_txt_data(dir="./data/ZuCo", model_type="BART"):
    with open(f"{dir}/ZuCoProcessedDatasetDict-{model_type}.pkl", "rb") as f:
        master_eeg = pickle.load(f)
    with open(f"{dir}/ZuCoTargetStringsEmbeds-{model_type}.pkl", "rb") as f:
        master_embeds = pickle.load(f)
    return {"data" : master_eeg, "targets" : master_embeds}

def load_img_data(dir="./data/Brain2Image"):
    with open(f"{dir}/imageNet_labeled_eeg.pkl", "rb") as f:
        image_eeg_labels = pickle.load(f)
    with open(f"{dir}/image_net_dict.pkl", "rb") as f:
        img_net_dict = pickle.load(f)
    with open(f"{dir}/label_dict.pkl", "rb") as f:
        label_dict = pickle.load(f)
    with open(f"{dir}/latent_dict.pkl", "rb") as f:
        latent_dict = pickle.load(f)
    return {"data" : image_eeg_labels, "targets" : img_net_dict, "labels" : label_dict, "latents" : latent_dict}
from transformers import CLIPTokenizer, CLIPTextModel
device="cuda:0"
bart_tokenizer = CLIPTokenizer.from_pretrained('facebook/bart-large')
clip_text_enc = CLIPTextModel.from_pretrained('facebook/bart-large').to(device)

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import pickle
import json
import matplotlib.pyplot as plt
from glob import glob
import time
import copy
from tqdm import tqdm
from data import *

whole_dataset_dicts = []
task_name = "task1_task2_taskNRv2"
if 'task1' in task_name:
    dataset_path_task1 = './data/ZuCo/task1-SR/task1-SR-dataset.pickle'
    with open(dataset_path_task1, 'rb') as handle:
        whole_dataset_dicts.append(pickle.load(handle))
if 'task2' in task_name:
    dataset_path_task2 = './data/ZuCo/task2-NR/task2-NR-dataset.pickle'
    with open(dataset_path_task2, 'rb') as handle:
        whole_dataset_dicts.append(pickle.load(handle))
if 'task3' in task_name:
    dataset_path_task3 = './data/ZuCo/task3-TSR/task3-TSR-dataset.pickle'
    with open(dataset_path_task3, 'rb') as handle:
        whole_dataset_dicts.append(pickle.load(handle))
if 'taskNRv2' in task_name:
    dataset_path_taskNRv2 = './data/ZuCo/task2-NR-2.0/task2-NR-2.0-dataset.pickle'
    with open(dataset_path_taskNRv2, 'rb') as handle:
        whole_dataset_dicts.append(pickle.load(handle))
        
# Process Raw ZuCO Dataset
subject_choice = "ALL"
eeg_type_choice = "GD"
bands_choice = ['_t1', '_t2', '_a1', '_a2', '_b1', '_b2', '_g1', '_g2']
dataset_setting = "unique_sent"
batch_size = 1
train_set = ZuCo_dataset(whole_dataset_dicts, 'train', clip_tokenizer, subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice, setting = dataset_setting)
dev_set = ZuCo_dataset(whole_dataset_dicts, 'dev', clip_tokenizer, subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice, setting = dataset_setting)
test_set = ZuCo_dataset(whole_dataset_dicts, 'test', clip_tokenizer, subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice, setting = dataset_setting)

master_set = {"train" : train_set, "dev" : dev_set, "test" : test_set}
with open ("./data/ZuCo/ZuCoProcessedDatasetDict-CLIP.pkl", "wb") as f:
    pickle.dump(master_set, f)
    
all_targets = {}
for key in ["train", "dev", "test"]:
    target_strings = []
    with torch.no_grad():
        for input_embeddings, seq_len, input_masks, input_mask_invert, target_ids, target_mask, sentiment_labels, sent_level_EEG in tqdm(master_set[key]):
            input_embeddings_batch = input_embeddings.to(device).float()
            input_masks_batch = input_masks.to(device)
            input_mask_invert_batch = input_mask_invert.to(device)
            target_ids_batch = target_ids.to(device)

            target_tokens = clip_tokenizer.convert_ids_to_tokens(target_ids_batch.tolist(), skip_special_tokens = True)
            target_string = clip_tokenizer.decode(target_ids_batch, skip_special_tokens = True)
            target_strings.append(target_string)
        all_targets[key] = target_strings
        
with open("./data/ZuCo/ZuCoTargetStrings-CLIP.pkl", "wb") as f:
    pickle.dump(all_targets, f)
    
def text_tok(prompts, maxlen=None):
    with torch.no_grad():
        if maxlen is None: maxlen = clip_tokenizer.model_max_length
        inp = clip_tokenizer(prompts, padding="max_length", max_length=maxlen, truncation=True, return_tensors="pt")
        return (inp.input_ids.to(device), inp.attention_mask.to(device))

def text_enc(prompts, maxlen=None):
    with torch.no_grad():
        if maxlen is None: maxlen = clip_tokenizer.model_max_length
        inp = clip_tokenizer(prompts, padding="max_length", max_length=maxlen, truncation=True, return_tensors="pt")
        return clip_text_enc(inp.input_ids.to(device))[0].half()
    
keys = ["train", "dev", "test"]
all_embeds = {}
for key in keys:
    cur_embeds = []
    for idx in tqdm(range(len(all_targets[key]))):
    # for idx in tqdm(range(10)):
    cur_embeds.append(text_enc(all_targets[key][idx]))
    all_embeds[key] = cur_embeds
    
with open ("./data/ZuCo/ZuCoTargetStringsEmbeds-CLIP.pkl", "wb") as f:
    pickle.dump(all_embeds, f)
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
import torch.nn.functional as F

PATH_zuco_dict = "./data/ZuCo/ZuCoProcessedDatasetDict.pkl"
PATH_zuco_embeds = "./data/ZuCo/ZuCoTargetStringsEmbeds.pkl"

# Load CLIP Models
from transformers import CLIPTokenizer, CLIPTextModel
device = "cuda:0" if torch.cuda.is_available() else "cpu"
clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
clip_text_enc = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device)


if __name__ == "__main__":
	
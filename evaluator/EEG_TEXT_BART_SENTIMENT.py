import sys
import time

sys.path.append("./models")
sys.path.append("./training")
sys.path.append("./testing")
sys.path.append("./utils")
sys.path.append("./ZuCo")

from master_init import *
from DSG import *
from load_data import *
from data import *
from dataloader import *
from count_params import *

import torch
import torch.nn as nn
import torch.optim as optim

from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig

import numpy as np
import pickle

def evaluate(args_dict, save_results_path = None):
    dataloader = args_dict["dataloader"]
    model = args_dict["model"]
    tokenizer = args_dict["tokenizer"]
    device = args_dict["device"] if "device" in args_dict else "cuda"
    device_ids = args_dict["device_ids"] if "device_ids" in args_dict else None
    staging_device = args_dict["staging_device"] if "staging_device" in args_dict else None

    if staging_device==None:
        staging_device = f"cuda:{device_ids[0]}" if not device_ids == None else "cuda"
    results = {}
    for phase in ['test']:
        # confusion_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        num_classes = 3  #there are 3 classes
        confusion_matrix = np.zeros((num_classes, num_classes))
        model.eval()     # Set model to evaluate mode

        # Iterate over data.
        current_data = dataloader[phase].load_data()
        while not current_data["reset"]:
            input_embeddings, seq_len, input_masks, input_mask_invert, target_ids, target_mask, sentiment_labels, sent_level_EEG = current_data["data"]

            good = []
            for i in range(len(sentiment_labels)):
                if not sentiment_labels[i] == -100:
                    good.append(i)
            
            if len(good) == 0:
                current_data = dataloader[phase].load_data()
                continue
            
            input_embeddings = input_embeddings[good]
            input_masks = input_masks[good]
            input_mask_invert = input_mask_invert[good]
            target_ids = target_ids[good]
            sentiment_labels = sentiment_labels[good]

            target = torch.zeros(input_embeddings.shape[0], 3).to(device)
            for i in range(input_embeddings.shape[0]):
                target[i][sentiment_labels[i]] = 1.0

            input_embeddings_batch = input_embeddings.to(staging_device).float()
            input_masks_batch = input_masks.to(staging_device)
            input_mask_invert_batch = input_mask_invert.to(staging_device)
            target_ids_batch = target_ids.to(staging_device)

            """replace padding ids in target_ids with -100"""
            target_ids_batch[target_ids_batch == tokenizer.pad_token_id] = -100

            model.zero_grad()

            args_dict = {
                "input_data_batch" : input_embeddings_batch,
                "input_masks_batch" : input_masks_batch,
                "input_masks_invert" : input_mask_invert_batch,
                "target_ids_batch" : target_ids_batch,
                "pool_result" : True,
                "temperature" : 0.04
                }

            output = model(
                mode="EEG-TEXT-BART-SENTIMENT",
                args_dict=args_dict,
                staging_device=staging_device
                )
            
            # for i in range(input_embeddings.shape[0]):
            #     confusion_matrix[sentiment_labels[i]][torch.argmax(output[i])] += 1
            for i in range(input_embeddings.shape[0]):
                true_label = sentiment_labels[i]
                predicted_label = torch.argmax(output[i])
                confusion_matrix[true_label][predicted_label] += 1
            
            current_data = dataloader[phase].load_data()
        
        # confusion_matrix = np.array(confusion_matrix)
        # results[f"{phase}_confusion_matrix"] = confusion_matrix
        # tp = np.array([confusion_matrix[0][0], confusion_matrix[1][1], confusion_matrix[2][2]])
        # fp = np.array([confusion_matrix[1][0] + confusion_matrix[2][0],
        # confusion_matrix[0][1] + confusion_matrix[2][1],
        # confusion_matrix[0][2] + confusion_matrix[1][2]])
        # fn = np.array([confusion_matrix[0][1] + confusion_matrix[0][2],
        # confusion_matrix[1][0] + confusion_matrix[1][2],
        # confusion_matrix[2][0] + confusion_matrix[2][1]])
        # tn = np.array([confusion_matrix[1][1] + confusion_matrix[1][2] + confusion_matrix[2][1] + confusion_matrix[2][2],
        # confusion_matrix[0][0] + confusion_matrix[0][2] + confusion_matrix[2][0] + confusion_matrix[2][2],
        # confusion_matrix[0][0] + confusion_matrix[0][1] + confusion_matrix[1][0] + confusion_matrix[1][1]])



        tp = np.diag(confusion_matrix)
        fp = np.sum(confusion_matrix, axis=0) - tp
        fn = np.sum(confusion_matrix, axis=1) - tp
#         tn = np.sum(confusion_matrix) - (fp + fn + tp)

        # 计算每个类别的precision, recall, accuracy, F1分数
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        # accuracy = (tp + tn) / (tp + fp + fn + tn)

        f1 = 2 * (precision * recall) / (precision + recall)



        # 计算整体指标（微平均和宏平均）
        micro_avg_precision = tp.sum() / (tp.sum() + fp.sum())
        micro_avg_recall = tp.sum() / (tp.sum() + fn.sum())
        micro_avg_f1 = 2 * (micro_avg_precision * micro_avg_recall) / (micro_avg_precision + micro_avg_recall)

        macro_avg_precision = precision.mean()
        macro_avg_recall = recall.mean()
        macro_avg_f1 = f1.mean()
        
        results[f"{phase}_TP"] = tp
        results[f"{phase}_FP"] = fp
        # results[f"{phase}_TN"] = tn
        results[f"{phase}_FN"] = fn
        results[f"{phase}_precision"] = precision = tp / (tp + fp)
        results[f"{phase}_recall"] = recall = tp / (tp + fn)
        results[f"{phase}_f1"] = f1 = 2 * (precision * recall) / (precision + recall)
        results[f"{phase}_macro_avg_precision"] = macro_avg_precision
        results[f"{phase}_macro_avg_recall"] = macro_avg_recall
        results[f"{phase}_macro_avg_f1"] = macro_avg_f1
        results[f"{phase}_micro_avg_precision"] = micro_avg_precision
        results[f"{phase}_micro_avg_recall"] = micro_avg_recall
        results[f"{phase}_micro_avg_f1"] = micro_avg_f1
    model.zero_grad()
    if not save_results_path == None:
        with open(save_results_path, "wb") as f:
            pickle.dump(results, f)
    return results

if __name__ == "__main__":
    # CKPT_DIR = "./checkpoints/layerwise"
    # RESULTS_DIR = "./results/Layerwise"
    # MODEL_NAME = "MMMM_BART+DIFFUSION+SENT_FINAL-3"
    CKPT_DIR = "./checkpoints/layerwise_pe"
    RESULTS_DIR = "./results/layerwise_pe"
    MODEL_NAME = "MMMM_ALL_FINAL"

    device="cuda"
    device_ids=[0,1,2,3]
    
    model = INITIALIZE_MODEL(device=device, device_ids=device_ids)
    model = nn.DataParallel(model, device_ids=device_ids).to(device)
    state_dict=torch.load(f"{CKPT_DIR}/{MODEL_NAME}.pt")
    model.load_state_dict(state_dict)
    print("[INFO] Loaded model checkpoint.")
    
    dataloaders = INITIALIZE_DATALOADERS(
        keys=["ZuCo-BART"],
        bsz=[1],
        dev_bsz=[1]
    )
    
    ZuCo_dataloader=dataloaders["ZuCo-BART"]
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    
    criterion = nn.CrossEntropyLoss()
    
    args_dict = {
        "dataloader":ZuCo_dataloader,
        "device":device,
        "tokenizer":tokenizer,
        "criterion":criterion,
        "model":model
    }
    
    results = evaluate(args_dict, save_results_path=f"{RESULTS_DIR}/Sentiment_{MODEL_NAME}_pe.pkl")
    print(results)
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
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from rouge import Rouge


def eval_model(dataloader, device, tokenizer, criterion, model, output_all_results_path = './Results/MMMM_EEG-TEXT-BART/all_decoding_results.txt', device_ids=None):
    staging_device = f"cuda:{device_ids[0]}" if device_ids == None else "cuda"
    model.eval()
    
    running_loss = 0.0
    sample_count = 0
    
    target_tokens_list = []
    target_string_list = []
    pred_tokens_list = []
    pred_string_list = []
    
    phase = "test"
    current_data = dataloader[phase].load_data()
    with open(output_all_results_path,'w') as f:
        while not current_data["reset"]:
            input_embeddings, seq_len, input_masks, input_mask_invert, target_ids, target_mask, sentiment_labels, sent_level_EEG = current_data["data"]
            
            # Load Batch
            input_embeddings_batch = input_embeddings.to(staging_device).float()
            input_masks_batch = input_masks.to(staging_device)
            target_ids_batch = target_ids.to(staging_device)
            input_mask_invert_batch = input_mask_invert.to(staging_device)

            target_tokens = tokenizer.convert_ids_to_tokens(target_ids_batch[0].tolist(), skip_special_tokens = True)
            target_string = tokenizer.decode(target_ids_batch[0], skip_special_tokens = True)
            f.write(f'target string: {target_string}\n')

            # Add for BLEU score later
            target_tokens_list.append([target_tokens])
            target_string_list.append(target_string)

            """replace padding ids in target_ids with -100"""
            target_ids_batch[target_ids_batch == tokenizer.pad_token_id] = -100 
            
            args_dict = {
                "input_data_batch" : input_embeddings_batch,
                "input_masks_batch" : input_masks_batch,
                "input_masks_invert" : input_mask_invert_batch,
                "target_ids_batch" : target_ids_batch,
                "pool_result" : False
                }
            
            seq2seqLMoutput = model(
                mode="EEG-TEXT-BART",
                args_dict=args_dict,
                staging_device=staging_device
            )
            
             # Use the BART language modeling loss
            loss = seq2seqLMoutput.loss

            logits = seq2seqLMoutput.logits # 8*48*50265
            probs = logits[0].softmax(dim=1)
            values, predictions = probs.topk(1)
            predictions = torch.squeeze(predictions)
            predicted_string = tokenizer.decode(predictions).split('</s></s>')[0].replace('<s>','')
            f.write(f'predicted string: {predicted_string}\n')
            f.write(f'################################################\n\n\n')

            # convert to int list
            predictions = predictions.tolist()
            truncated_prediction = []
            for t in predictions:
                if t != tokenizer.eos_token_id:
                    truncated_prediction.append(t)
                else:
                    break
            pred_tokens = tokenizer.convert_ids_to_tokens(truncated_prediction, skip_special_tokens = True)
            pred_tokens_list.append(pred_tokens)
            pred_string_list.append(predicted_string)
            
            # Compute Statistics
            sample_count += 1
            running_loss += loss.item() * input_embeddings_batch.shape[0] # batch loss
            current_data = dataloader[phase].load_data()

        epoch_loss = running_loss / sample_count
        print('[INFO] test loss: {:4f}'.format(epoch_loss))
        f.write('[INFO] test loss: {:4f}\n'.format(epoch_loss))

        """ calculate corpus bleu score """
        weights_list = [(1.0,),(0.5,0.5),(1./3.,1./3.,1./3.),(0.25,0.25,0.25,0.25)]
        for weight in weights_list:
            # print('weight:',weight)
            corpus_bleu_score = corpus_bleu(target_tokens_list, pred_tokens_list, weights = weight)
            print(f'[INFO] corpus BLEU-{len(list(weight))} score:', corpus_bleu_score)
            f.write(f'[INFO] corpus BLEU-{len(list(weight))} score: {corpus_bleu_score}\n')

        print()
        f.write('\n')
        """ calculate rouge score """
        rouge = Rouge()
        rouge_scores = rouge.get_scores(pred_string_list,target_string_list, avg = True)
        print(rouge_scores)
        f.write(f"{rouge_scores}\n")


if __name__ == '__main__': 
    CKPT_DIR = "./checkpoints/SimpleRoundRobin"
    RESULTS_DIR = "./results/SimpleRoundRobin"
    MODEL_NAME = "MMMM_50"
    
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
    print("[INFO] Intialized dataloaders.")
    
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    
    criterion = nn.CrossEntropyLoss()
    
    ''' eval '''
    eval_model(ZuCo_dataloader, device, tokenizer, criterion, model, output_all_results_path=f"{RESULTS_DIR}/all_decoding_results_{MODEL_NAME}.txt", device_ids=device_ids)

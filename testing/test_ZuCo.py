import torch

def test_ZuCo(test_dataloader, model, loss_fn, device="cuda:0"):
  zuco_dataloader = test_dataloader
  eeg_enc = model
  criterion = loss_fn

  cur_loss = 0.0
  tot_cnt = 0

  zuco_data = zuco_dataloader.load_data()
  while not zuco_data["reset"]:
      input_embeddings, seq_len, input_masks, input_mask_invert, target_ids, target_mask, sentiment_labels, sent_level_EEG = zuco_data["data"]
      res = eeg_enc("TXT", input_embeddings.to(device).float(), input_masks.to(device), input_mask_invert.to(device))
      embed = zuco_data["target"]

      loss = criterion(res.to(device).float().view(embed.shape[0] * 77, 768), embed.to(device).float().view(embed.shape[0] * 77, 768), torch.ones(embed.shape[0] * 77).to(device))
      eeg_enc.zero_grad()

      cur_loss += loss.item() * zuco_data["size"]
      tot_cnt += zuco_data["size"]
      zuco_data = zuco_dataloader.load_data()

  cur_loss /= tot_cnt
  return {"loss" : cur_loss}
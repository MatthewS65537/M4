def test_Brain2Image(test_dataloader, model, loss_fn):
  image_net_dataloader = test_dataloader
  eeg_enc = model
  loss = loss_fn

  cur_loss = 0.0
  tot_cnt = 0

  image_net_data = image_net_dataloader.load_data()
  while not image_net_dataloader.reset():
      input_data_batched = image_net_data["data"]
      input_data_batched_converted = torch.zeros(tuple([len(input_data_batched)]) + input_data_batched[0].shape).to(device)
      for i in range(len(input_data_batched)):
        input_data_batched_converted[i] = input_data_batched[i].to(device)
      target_batched = image_net_data["target"]
      target_batched_converted = torch.zeros(tuple([len(target_batched)]) + target_batched[0].shape).to(device)
      for i in range(len(target_batched)):
        target_batched_converted[i] = target_batched[i].to(device)

      res = eeg_enc("IMG", input_data_batched_converted.to(device).float(), pool_img_head=True)
      loss = criterion(res.to(device).float().view(target_batched_converted.shape[0], 768),
                        target_batched_converted.to(device).float().view(target_batched_converted.shape[0], 768),
                        torch.ones(target_batched_converted.shape[0] * 77).to(device))

      eeg_enc.zero_grad()

      cur_loss += loss.item() * image_net_data["size"]
      tot_cnt += image_net_data["size"]
      image_net_data = image_net_dataloader.load_data()

    cur_loss /= tot_cnt

    return {"loss" : cur_loss}
import sys

sys.path.append("./models")
sys.path.append("./training")
sys.path.append("./utils")

from eeg_encoder import *
from DSG import *
from load_data import *

# if __name__ == "__main__":
master_eeg, master_embeds = load_txt_data()
image_eeg_labels, img_net_dict = load_img_data()

zuco_dataloader = ZuCoDataloader(master_eeg["train"], master_embeds["train"], bsz=64, drop_last=True)
image_net_dataloader = ImageNetDataloader(image_eeg_labels, img_net_dict, bsz=1, drop_last=True)

dsg_tasks = DSGTasks()
dsg_tasks.add_task(DSGTask("EEG-TXT", dataset=zuco_dataloader, converge_lim=10, div_threshold=0.01))
dsg_tasks.add_task(DSGTask("EEG-IMG", dataset=image_net_dataloader, converge_lim=10, div_threshold=0.01))

import torch.nn
import torch.nn.optim

device = "cuda:0"

eeg_enc = EEGEncoder(txt_in_feat=840, img_in_feat=500, enc_feat=1024, dec_emb_sz=768, enc_nhead=8, enc_dim_ff=2048, num_enc_layers=8)

criterion = nn.CosineEmbeddingLoss()
optimizer = optim.Adam(eeg_enc.parameters(), lr=learning_rate)

epoch_num = 0
while dsg_tasks.should_keep_training():
	epoch_num += 1
	for task in dsg_tasks:
		if task.should_keep_training():
			cur_loss = 0.0
			if task_name == "EEG-TXT":

			if task_name == "EEG-IMG":


			task.update(epoch_num, cur_loss)
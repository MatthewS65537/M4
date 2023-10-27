import torch
from torch.utils.data import DataLoader

class ImageNetDataloader():
	def __init__(self, labeled_eeg, image_net_dict, bsz=64, drop_last = True):
		super(ImageNetDataloader, self).__init__()
		self.labeled_eeg = labeled_eeg
		self.image_net_dict = image_net_dict
		self.bsz = bsz
		self.ptr = 0
		self.dataset_size = len(labeled_eeg["labels"])
		self.drop_last = drop_last
		self.reset = False

	def load_data():
		self.ptr += bsz
		if (self.ptr > (dataset_size // bsz * bsz) and drop_last) or (self.ptr > dataset_size and not drop_last):
			self.reset = True
		target = [img_dict[labeled_eeg["labels"][i]] for i in range(ptr - 64, min(ptr, dataset_size))]
		sz = min(ptr, dataset_size) - (ptr - 64)
		return {"data" : self.labeled_eeg["eeg"][ptr-bsz:min(ptr, dataset_size)], "target" : target, "size" : sz}

	def reset():
		if self.reset:
			self.reset = False
			self.ptr = 0
			return True
		return False



class ZuCoDataloader():
	def __init__(self, data, targets, bsz=64, drop_last = True):
		super(ImageNetDataloader, self).__init__()
		self.eeg_dataloader = DataLoader(data, batch_size=bsz, shuffle=False, drop_last=drop_last)
		self.embeds_dataloader = DataLoader(targets, batch_size=bsz, shuffle=False, drop_last=drop_last)
		self.eeg_iter = iter(eeg_dataloader)
		self.embeds_iter = iter(embeds_dataloader)
		self.ptr = 0
		self.bsz = bsz
		self.data = 

	def load_data():
		eeg = next(eeg_iter)
		embed = next(embeds_iter)[:,0,:,:]
		ptr += bsz

		return {"data" : eeg, "target" : embed, "size" : embed.shape[0]}


	def reset():
		self.eeg_iter = iter(eeg_dataloader)
		self.embeds_iter = iter(embeds_dataloader)
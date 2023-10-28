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
		self.is_reset = False

	def load_data(self):
		self.ptr += self.bsz
		if (self.ptr > (self.dataset_size // self.bsz * self.bsz) and self.drop_last) or (self.ptr > self.dataset_size and not self.drop_last):
			self.is_reset = True
		target = [self.image_net_dict[self.labeled_eeg["labels"][i]] for i in range(self.ptr - 64, min(self.ptr, self.dataset_size))]
		sz = min(self.ptr, self.dataset_size) - (self.ptr - 64)
		return {"data" : self.labeled_eeg["eeg"][self.ptr-self.bsz:min(self.ptr, self.dataset_size)], "target" : target, "size" : sz}

	def reset(self):
		if self.is_reset:
			self.is_reset = False
			self.ptr = 0
			return True
		return False

class ZuCoDataloader():
	def __init__(self, data, targets, bsz=64, drop_last = True):
		super(ZuCoDataloader, self).__init__()
		self.eeg_dataloader = DataLoader(data, batch_size=bsz, shuffle=False, drop_last=drop_last)
		self.embeds_dataloader = DataLoader(targets, batch_size=bsz, shuffle=False, drop_last=drop_last)
		self.eeg_iter = iter(self.eeg_dataloader)
		self.embeds_iter = iter(self.embeds_dataloader)

	def load_data(self):
		try:
			eeg = next(self.eeg_iter)
			embed = next(self.embeds_iter)[:,0,:,:]

			return {"data" : eeg, "target" : embed, "size" : embed.shape[0]}
		except StopIteration:
			self.reset()
			self.load_data()
			return "RESET"
	def reset(self):
		self.eeg_iter = iter(self.eeg_dataloader)
		self.embeds_iter = iter(self.embeds_dataloader)
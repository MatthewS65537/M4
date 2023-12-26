import torch
from torch.utils.data import DataLoader

class ImageNetDataloader():
    def __init__(self, labeled_eeg, image_net_dict, label_dict, bsz=1, drop_last=True):
        super(ImageNetDataloader, self).__init__()
        self.labeled_eeg = labeled_eeg
        self.image_net_dict = image_net_dict
        self.label_dict = label_dict
        self.bsz = bsz
        self.ptr = 0
        self.dataset_size = len(labeled_eeg["labels"])
        self.drop_last = drop_last
        self.is_reset = False

        if not self.bsz == 1:
            print("[WARNING] IMAGE NET TRAINING NOT IMPLEMENTED FOR BSZ NOT 1. PROCEED WITH CAUTION.")

    def load_data(self):
        self.is_reset = False
        self.ptr += self.bsz
        if (self.ptr > (self.dataset_size // self.bsz * self.bsz) and self.drop_last) or (self.ptr > self.dataset_size and not self.drop_last):
            self.reset()
            self.ptr += self.bsz
        
        labels = [self.labeled_eeg["labels"][i] for i in range(self.ptr - self.bsz, min(self.ptr, self.dataset_size))]
        classes = [self.label_dict[i.split("_")[0]] for i in labels]
        target = [self.image_net_dict[self.labeled_eeg["labels"][i]] for i in range(self.ptr - self.bsz, min(self.ptr, self.dataset_size))]
        sz = min(self.ptr, self.dataset_size) - (self.ptr - self.bsz)
        data = self.labeled_eeg["eeg"][self.ptr-self.bsz:min(self.ptr, self.dataset_size)]
        data = data[0].unsqueeze(0)
        if not self.bsz == 1:
            print("[WARNING] IMAGE NET TRAINING NOT IMPLEMENTED FOR BSZ NOT 1. AUTOMATICALLY DROPPED ALL BUT FIRST IN BATCH.")
        return {"data" : data, "target" : target, "size" : sz, "reset" : self.is_reset, "labels" : labels, "classes" : classes}

    def query_data(self):
        return self.image_net_dict

    def query_labels(self):
        return self.labeled_eeg["labels"]

    def reset(self):
        self.ptr = 0
        self.is_reset = True

    def set_bsz(self, new_bsz):
        self.bsz = new_bsz

class ZuCoDataloader():
    def __init__(self, data, targets, bsz=64, drop_last=True):
        super(ZuCoDataloader, self).__init__()
        self._data = data
        self._targets = targets
        self._drop_last = drop_last
        self.eeg_dataloader = DataLoader(data, batch_size=bsz, shuffle=False, drop_last=drop_last)
        self.embeds_dataloader = DataLoader(targets, batch_size=bsz, shuffle=False, drop_last=drop_last)
        self.eeg_iter = iter(self.eeg_dataloader)
        self.embeds_iter = iter(self.embeds_dataloader)
        self.bsz = bsz

    def load_data(self, just_reset=False):
        try:
            eeg = next(self.eeg_iter)
            embed = next(self.embeds_iter)[:,0,:,:]

            return {"data" : eeg, "target" : embed, "size" : embed.shape[0], "reset" : just_reset}
        except StopIteration:
            self.reset()
            return self.load_data(just_reset=True)

    def reset(self):
        self.eeg_iter = iter(self.eeg_dataloader)
        self.embeds_iter = iter(self.embeds_dataloader)

    def set_bsz(self, new_bsz):
        self.bsz = new_bsz
        self.eeg_dataloader = DataLoader(self._data, batch_size=new_bsz, shuffle=False, drop_last=self._drop_last)
        self.embeds_dataloader = DataLoader(self._targets, batch_size=new_bsz, shuffle=False, drop_last=self._drop_last)
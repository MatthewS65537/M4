import torch
from torch.utils.data import DataLoader

class ImageNetDataloader():
    def __init__(self, labeled_eeg, image_net_dict, bsz=1, drop_last=True):
        super(ImageNetDataloader, self).__init__()
        self.labeled_eeg = labeled_eeg
        self.image_net_dict = image_net_dict
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
        target = [self.image_net_dict[self.labeled_eeg["labels"][i]] for i in range(self.ptr - self.bsz, min(self.ptr, self.dataset_size))]
        sz = min(self.ptr, self.dataset_size) - (self.ptr - self.bsz)
        return {"data" : self.labeled_eeg["eeg"][self.ptr-self.bsz:min(self.ptr, self.dataset_size)], "target" : target, "size" : sz, "reset" : self.is_reset, "labels" : labels}

    def query_data(self):
        return self.image_net_dict

    def query_labels(self):
        return self.labeled_eeg["labels"]

    def reset(self):
        self.ptr = 0
        self.is_reset = True

class ZuCoDataloader():
    def __init__(self, data, targets, bsz=64, drop_last=True):
        super(ZuCoDataloader, self).__init__()
        self.eeg_dataloader = DataLoader(data, batch_size=bsz, shuffle=False, drop_last=drop_last)
        self.embeds_dataloader = DataLoader(targets, batch_size=bsz, shuffle=False, drop_last=drop_last)
        self.eeg_iter = iter(self.eeg_dataloader)
        self.embeds_iter = iter(self.embeds_dataloader)

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
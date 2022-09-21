import torch, os, torch.nn as nn, pytorch_lightning as pl, numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

class HSQCDataset(Dataset):
    def __init__(self, split="train", tessellation=False, voronoi=False,voronoi_pos_neg=False):
        self.dir = "/root/data/hyun_fp_data/hsqc_ms_pairs/"
        self.tessellation=tessellation
        self.voronoi=voronoi
        self.voronoi_pos_neg=voronoi_pos_neg
        
        self.split = split
        # self.orig_hsqc = os.path.join(self.dir, "data")
        # assert(os.path.exists(self.orig_hsqc))
        assert(split in ["train", "val", "test"])
        self.FP_files = list(os.listdir(os.path.join(self.dir, split, "FP")))
        # self.HSQC_files = list(os.listdir(os.path.join(self.dir, split, "HSQC")))
        # assert (len(self.FP_files ) == (self.HSQC_files))

    def __len__(self):
        return len(self.FP_files)//2

            
    def __getitem__(self, i):
        index = i*2
        # hsqc = torch.load(os.path.join(self.dir,  self.split, "HSQC_plain_imgs", self.FP_files[i]))
        hsqc1 = torch.load(os.path.join(self.dir,  self.split, "HSQC_sign_only", self.FP_files[index]))
        hsqc2 = torch.load(os.path.join(self.dir,  self.split, "HSQC_sign_only", self.FP_files[index+1]))
        hsqc1 = torch.abs(hsqc1)
        hsqc2 = torch.abs(hsqc2)
        hsqc_overlap1 = hsqc1*2 + hsqc2
        hsqc_overlap2 = hsqc1 + hsqc2*2
        hsqc_display = torch.sign(hsqc_overlap1)
        # hsqc_overlap1 = F.one_hot(hsqc_overlap1.to(torch.int64), num_classes = 4).permute(0, 3, 1, 2)
        # hsqc_overlap2 = F.one_hot(hsqc_overlap2.to(torch.int64), num_classes = 4).permute(0, 3, 1, 2)
        #     # will be shape of torch.Size([1, 180, 120, 4])
        return hsqc_display.float(), torch.squeeze(hsqc_overlap1.long(),1), torch.squeeze(hsqc_overlap2.long(),1)

# def pad(batch):
#     hsqc, fp = zip(*batch)
#     hsqc = pad_sequence([v.clone().detach() for v in hsqc], batch_first=True)
#     return hsqc, torch.stack(fp)

class HsqcDataModule(pl.LightningDataModule):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size

    
    def setup(self, stage):
        if stage == "fit" or stage is None:
            self.train = HSQCDataset("train")
            self.val = HSQCDataset("val")
        if stage == "test":
            self.test = HSQCDataset("test")
        if stage == "predict":
            raise NotImplementedError("Predict setup not implemented")

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=4, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=4)


# loader = DataLoader(HSQCDataset(split="test"), batch_size=32, collate_fn=pad, num_workers=4)
# for iter, data in enumerate(loader):
#     hsqc, fp = data
#     break
from locale import normalize
import pytorch_lightning as pl
import torch, torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

import torchvision.models as models
import numpy as np
import copy,sys

from torchvision.models.resnet import BasicBlock, Bottleneck
sys.path.append("/root")
from utils.customized_weighted_BCElosswithlogit import Customiezed_BCEWithLogits
from utils.unet_parts import *



class UNet(pl.LightningModule):
    def __init__(self, config):
        super(UNet, self).__init__()
        n_channels_in = 1
        n_channels_out = 4
        bilinear = config['bilinear']
        tessellation=False

        self.inc = DoubleConv(n_channels_in, 64)
        if tessellation:
            self.down1 = Down(128,128)
        else:
            self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_channels_out)
        
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, tessellate_info=None, feature=False):
        x1 = self.inc(x)
        if tessellate_info!=None:
            concatenated = torch.concat((x1, tessellate_info), 1)
            x2 = self.down1(concatenated)
        else:
            x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # print('x5 shape is ', x5.shape)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        if feature:
            return logits, x5
        return logits
    

    def training_step(self, batch, batch_idx):
        hsqc_display, hsqc_overlap1, hsqc_overlap2 = batch
        out = self(hsqc_display) * hsqc_display # only keep the value where there were dots
        print (out.shape)
        _, out = torch.max(out, 1)
        print(out.shape)
        loss, accu = self.compute_loss_and_accu(out,  hsqc_overlap1, hsqc_overlap2)
        self.log("tr/loss", loss)
        return loss

    def training_epoch_end(self, train_step_outputs):
        mean_loss = sum([t["loss"].item() for t in train_step_outputs]) / len(train_step_outputs)
        self.log("tr/mean_loss", mean_loss)

    def validation_step(self, batch, batch_idx):
        x, labels = batch
        out = self.forward(x)
        loss = self.loss(out, labels)
        metrics = self.compute_metrics(out, labels, self.ranker, loss)
        metrics["ce_loss"]=loss.item()
        # for k,v in metrics.items():
        #     self.log(k, v)
        return metrics
    
    def test_step(self, batch, batch_idx):
        x, labels = batch
        out = self.forward(x)
        loss = self.loss(out, labels)
        metrics = self.compute_metrics(out, labels, self.ranker, loss)
        for k,v in metrics.items():
            self.log(k, v)
        return metrics
    

    def validation_epoch_end(self, validation_step_outputs):
        # return None
        feats = validation_step_outputs[0].keys()
        di = {}
        for feat in feats:
            di[f"val/mean_{feat}"] = np.mean([v[feat] for v in validation_step_outputs])
        for k,v in di.items():
            self.log(k, v)

    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr = self.lr)

    def init_weights(self):
        # refer to https://github.com/pytorch/vision/blob/a75fdd4180683f7953d97ebbcc92d24682690f96/torchvision/models/resnet.py#L160
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
        for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
                    
    # def configure_optimizers(self):
    #     # print(self.parameters())
    #     optimizer = torch.optim.SGD(self.parameters(),
    #                                 lr=self.lr,
    #                                 momentum=0.9, 
    #                                 weight_decay=0.0005)
    #     return {
    #         "optimizer": optimizer,
    #         "lr_scheduler": {
    #             "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min'),
    #             "monitor": 'val/mean_cos'
    #         },
    #     }
    
    def compute_loss_and_accu(self, out, hsqc_overlap1, hsqc_overlap2):
        loss1 = self.loss(out, hsqc_overlap1)
        loss2 = self.loss(out, hsqc_overlap2)
        if loss1 < loss2:
            loss = loss1 
            non_zero = hsqc_overlap1
            
            accu = torch.sum( (out==hsqc_overlap1))
        else:
            loss = loss2


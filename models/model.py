from torch import nn
import torch.nn.functional as F
from config import Config
import timm
from pytorch_lightning import LightningModule
import torch
from torchmetrics.functional import accuracy
from torchmetrics import F1
import pandas as pd
from datetime import datetime
from utils.utils import mixup_data, mixup_criterion
from torchvision import transforms

class FDModel(nn.Module):
    def __init__(self, cfg:Config):
        super(FDModel, self).__init__()
        self.cfg = cfg
        self.cnn = timm.create_model(
            cfg.model_name,
            pretrained=True,
            num_classes = 7,
            in_chans = 3
        )
    
    def forward(self, x):
        out = self.cnn(x)
        return out

class FDModule(LightningModule):
    def __init__(self, cfg:Config, class_weight=None):
        super().__init__()
        self.model = FDModel(cfg)
        self.val_metric = F1(num_classes=7, average="macro").cuda()
        self.train_metric =  F1(num_classes=7, average="macro").cuda()
        self.lr = 1e-4
        self.class_weight = class_weight
        self.cfg = cfg
        self.softmax = torch.nn.Softmax(dim=1)
        self.horizontalflip = transforms.RandomHorizontalFlip(p=1)
        self.verticalflip = transforms.RandomVerticalFlip(p=1)
        self.rotation_left = transforms.RandomRotation(degrees=(-90,-90))
        self.rotation_right = transforms.RandomRotation(degrees=(90,90))

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        if batch_idx % 4 == 0:
            mixed_x, y_a, y_b, lam = mixup_data(x, y)
            logits = self(mixed_x)
            loss = mixup_criterion(F.cross_entropy, logits, y_a, y_b, lam, torch.Tensor(self.class_weight).cuda())
            self.log_dict({'mixup_loss':loss})
            return loss
        logits = self(x)
        loss = F.cross_entropy(logits, y.long(), weight= torch.Tensor(self.class_weight).cuda())
        preds = torch.argmax(logits, dim=1)
        micro_acc = accuracy(preds, y)
        
        f1_score = self.train_metric(preds, y)
        self.log_dict(
            {
                "train_loss": loss,
                "train_acc": micro_acc,
                "train_f1_macro": f1_score
            },
            prog_bar=True,
            sync_dist=True,
            on_step=False,
            on_epoch=True

        )
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y.long(), weight= torch.Tensor(self.class_weight).cuda())

        preds = torch.argmax(logits, dim=1)
        micro_acc = accuracy(preds, y)

        f1_score = self.val_metric(preds, y)
        self.log_dict(
            {
                "val_loss": loss,
                "val_acc": micro_acc,
                "val_f1_macro": f1_score                
            },
            prog_bar=True,
            sync_dist=True,
            on_step=False,
            on_epoch=True
        )

    def test_step(self, batch, batch_idx):
        if self.cfg.tta:
            return self.tta(batch,batch_idx)
        x, uid = batch
        logits = self(x)
        #! Label Predict
        # preds = torch.argmax(logits, dim=1)
        #! Prob Predict
        # prob = self.softmax(logits)
        # preds = torch.max(prob,dim=1)
        #! Ensemble Predict
        prob = self.softmax(logits)
        preds = prob

        return preds, uid

    def tta(self, batch, batch_idx):
        x, uid = batch
        _normal = self.softmax(self(x))
        _h_flip = self.softmax(self(self.horizontalflip(x)))
        _v_flip = self.softmax(self(self.verticalflip(x)))
        _l_rotate = self.softmax(self(self.rotation_left(x)))
        _r_rotate = self.softmax(self(self.rotation_right(x)))
        preds = (_normal + _h_flip + _v_flip + _l_rotate + _r_rotate) / 5
        return preds, uid  

    def test_epoch_end(self, outputs):
        results = self.all_gather(outputs)
        df = pd.DataFrame(range(20000,24750),columns=['uid'])

        #! Prob for ensemble
        df['prob_0'] = -100.0
        df['prob_1'] = -100.0
        df['prob_2'] = -100.0
        df['prob_3'] = -100.0
        df['prob_4'] = -100.0
        df['prob_5'] = -100.0
        df['prob_6'] = -100.0

        
        df = df.set_index('uid')
        for p, u in results:
            #! Label
            # p = p.reshape(-1).cpu().numpy()
            # u = u.reshape(-1).cpu().numpy()
            # for pp, uu in zip(p,u):
            #     df.loc[uu] = pp

            #! Prob for Pseudo labeling
            # prob = p.values.reshape(-1).cpu().numpy()
            # l = p.indices.reshape(-1).cpu().numpy()
            # u = u.reshape(-1).cpu().numpy()
            # for pp, uu, ll in zip(prob,u,l):
            #     df.loc[uu] = [pp, ll]

            #! Prob for ensemble
            prob = p.reshape(-1,7).cpu().numpy()
            u = u.reshape(-1).cpu().numpy()
            for pp, uu in zip(prob,u):
                df.loc[uu] = pp
        df.to_csv(f'result_{self.cfg.exp}_{datetime.now().strftime("%d_%H_%M")}.csv')

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        #! org Adam
        # optimizer = Ranger21(self.model.parameters(), lr=1e-4, num_epochs=500, num_batches_per_epoch=126)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=1, eta_min=1e-9)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=200, T_mult=1)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=30, verbose=True)

        return [optimizer], [scheduler]
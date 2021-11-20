import os
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import StratifiedKFold
from torchvision import transforms

from config import Config
import cv2

import albumentations as A
import albumentations.pytorch as Ap

class FDDataset(Dataset):
    def __init__(self, cfg:Config, df:pd.DataFrame, aug:bool = True):
        super(FDDataset, self).__init__()
        self.cfg = cfg
        self.df = df
        self.aug = aug
        if self.aug:
            self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((256,256)),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomAffine(
                degrees=(-90,90),
                translate=(0.2, 0.2),
                scale=(0.8, 1.2), shear=15
                ),
                # transforms.RandomRotation(degrees=90),
                # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            ]
            )
        else:
            self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                transforms.Resize((256,256)),
            ]
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.cfg.data_dir, self.df.loc[idx, 'img_path'])
        
        img = cv2.imread(img_path)
        img = self.transform(img)
        if self.cfg.phase == 'test':
            return img, self.df.loc[idx, 'uid']
        label = self.df.loc[idx, 'disease_code']
        return img, label
        
class FDDataModule(LightningDataModule):
    def __init__(self, cfg:Config):
        super().__init__()
        self.cfg = cfg
        
        self.test_df = pd.read_csv(os.path.join(cfg.data_dir, 'test.csv'))
        self.train_df = pd.read_csv(os.path.join(cfg.data_dir, 'train.csv'))
        # self.train_df = pd.read_csv(os.path.join(cfg.data_dir, 'full_train.csv'))
        # self.val_df = pd.read_csv(os.path.join(cfg.data_dir,'val.csv'))
        self.fold_num = 0
        self._split_kfold()

    def set_fold_num(self, fold_num):
        self.fold_num = fold_num

    def get_class_weight(self):
        # return 1 / self.train_fold_df['disease_code'].value_counts().sort_index().values
        return 1 / self.train_df['disease_code'].value_counts().sort_index().values

    def setup(self, stage=None):
        if stage != 'test':
            print(f'FOLD NUM:{self.fold_num}')
            train_df = self.train_df[
                self.train_df["kfold"] != self.fold_num
            ].reset_index(drop=True)
            val_df = self.train_df[
                self.train_df["kfold"] == self.fold_num
            ].reset_index(drop=True)

            # train_df.to_csv('check_train_df.csv')
            # val_df.to_csv('check_val_df.csv')
            
            self.train = FDDataset(self.cfg, train_df, aug=True)
            self.val = FDDataset(self.cfg, val_df, aug=False)

            #! Test Data <-> Train Data
            # self.train = FDDataset(self.cfg, self.train_df.reset_index(drop=True))
            # self.val = FDDataset(self.cfg, self.val_df.reset_index(drop=True))

            self.train_fold_df = self.train_df

        if stage == 'test':
            self.test = FDDataset(self.cfg, self.test_df, aug=False)

    def _split_kfold(self):
        skf = StratifiedKFold(
            n_splits=Config.fold_num, shuffle=True, random_state=Config.seed
        )
        # (train_idx, val_idx)
        for n, (_, val_index) in enumerate(
            skf.split(
                X=self.train_df,
                y=self.train_df['disease_code']
            )
        ):  # if valid index, record fold num in 'kfold' column
            self.train_df.loc[val_index, "kfold"] = int(n)
        

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            shuffle=True,
            pin_memory=True,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            shuffle=False,
            pin_memory=True,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            shuffle=False,
        )

if __name__ == '__main__':
    dm = FDDataModule(Config)
    dm.setup()
    train_gen = dm.train_dataloader()
    d = iter(train_gen).next()
    print(d[0].shape) # B 3 512 512
    print(d[1].shape) # 128
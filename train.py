from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning import seed_everything
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from config import Config
from models.model import FDModule
from dataset import FDDataModule


seed_everything(Config.seed)

def train(cfg: Config, fold_num):
    
    fd_data_module = FDDataModule(cfg)
    # fd_data_module.setup(stage='test')
    
    #! TRAIN
    fd_data_module.set_fold_num(fold_num)
    fd_data_module.setup()
    class_weight = fd_data_module.get_class_weight()

    if cfg.phase=='test':
        fd_module = FDModule(cfg, class_weight=None).load_from_checkpoint(cfg.ckpt,
         cfg=Config)
    else:
        fd_module = FDModule(cfg, class_weight=class_weight)
    

    model_checkpoint = ModelCheckpoint(monitor='val_loss', save_top_k=3, dirpath=f'results/{cfg.exp}/{fd_data_module.fold_num}',
    filename="{epoch:02d}-{val_loss:.6f}-{val_acc:.4f}-{val_f1_macro}.pth", mode='min')

    early_stopping = EarlyStopping(monitor='val_loss', patience=200, verbose=True, mode='min') # for pseudo labeling
    # early_stopping = EarlyStopping(monitor='val_loss', patience=100, verbose=True, mode='min') # for full train

    trainer = pl.Trainer(
        gpus="0",
        accelerator='dp',
        num_nodes=1,
        deterministic=True,
        check_val_every_n_epoch=1,
        callbacks = [model_checkpoint, early_stopping],
        precision=16,
        log_every_n_steps=4,
        max_epochs = 500,
        auto_lr_find=True,
        plugins=DDPPlugin(find_unused_parameters=False),
    )
    
    if cfg.phase == 'train':
        trainer.fit(fd_module, fd_data_module)
    else:
        trainer.test(fd_module, fd_data_module)

if __name__ == '__main__':
    # train(Config, 0)
    # train(Config, 1)    
    # train(Config, 2)
    train(Config, 3)
    # train(Config, 4)
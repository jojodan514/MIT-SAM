import torch
from torch.utils.data import DataLoader
from utils.dataset import QaTa
from utils.dataset_mosmed import MosMed
import utils.config as config
from torch.optim import lr_scheduler
from engine.wrapper import LanGuideMedSegWrapper
import pytorch_lightning as pl    
from torchmetrics import Accuracy,Dice
from torchmetrics.classification import BinaryJaccardIndex
from pytorch_lightning.callbacks import ModelCheckpoint,EarlyStopping
from torch.backends import cudnn
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import argparse
import os
import random
import numpy as np
from pytorch_lightning import seed_everything

def seed_torch(seed=1029):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) 
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True


os.environ['CUDA_VISIBLE_DEVICES'] = '5,7'

def get_parser():
    parser = argparse.ArgumentParser(
        description='Language-guide Medical Image Segmentation')
    parser.add_argument('--config',
                        default='./config/training.yaml',
                        type=str,
                        help='config file')

    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)

    return cfg


if __name__ == '__main__':
    seed_torch()
    seed_everything(1029, workers=True)
    args = get_parser()
    print("cuda:",torch.cuda.is_available())

    # ds_train = QaTa(csv_path=args.qata_train_csv_path,
    #                 root_path=args.qata_train_root_path,
    #                 tokenizer=args.bert_type,
    #                 image_size=args.image_size,
    #                 mode='train')

    # ds_valid = QaTa(csv_path=args.qata_train_csv_path,
    #                 root_path=args.qata_train_root_path,
    #                 tokenizer=args.bert_type,
    #                 image_size=args.image_size,
    #                 mode='valid')

    ds_train = MosMed(
                    csv_path=args.mosmed_train_csv_path,
                    root_path=args.mosmed_train_root_path,
                    tokenizer=args.bert_type,
                    image_size=args.image_size,
                    mode='train')
    
    ds_valid = MosMed(csv_path=args.mosmed_test_csv_path,
                    root_path=args.mosmed_test_root_path,
                    tokenizer=args.bert_type,
                    image_size=args.image_size,
                    mode='valid')


    dl_train = DataLoader(ds_train, batch_size=args.train_batch_size, shuffle=True, num_workers=args.train_batch_size)
    dl_valid = DataLoader(ds_valid, batch_size=args.valid_batch_size, shuffle=False, num_workers=args.valid_batch_size)

    model = LanGuideMedSegWrapper(args)

    ## 1. setting recall function
    model_ckpt = ModelCheckpoint(
        dirpath=args.model_save_path,
        filename=args.model_save_filename,
        monitor='val_loss',
        save_top_k=1,
        mode='min',
        verbose=True,
    )

    early_stopping = EarlyStopping(monitor = 'val_loss',
                            patience=args.patience,
                            mode = 'min'
    )

    ## 2. setting trainer

    trainer = pl.Trainer(logger=True,
                        min_epochs=args.min_epochs,max_epochs=args.max_epochs,
                        accelerator='gpu', 
                        devices=args.device,
                        strategy="dp",
                        callbacks=[model_ckpt,early_stopping],
                        enable_progress_bar=False,
                        deterministic=True
                        ) 

    ## 3. start training
    print('start training')
    trainer.fit(model,dl_train,dl_valid)
    print('done training')


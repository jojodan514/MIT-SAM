import argparse
from engine.wrapper import LanGuideMedSegWrapper
from utils.dataset_mosmed import MosMed
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import pytorch_lightning as pl  

from utils.dataset import QaTa
import utils.config as config
import os
import random
import numpy as np
from pytorch_lightning import seed_everything

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

def seed_torch(seed=1029):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) 
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # 
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True


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

    # load model
    model = LanGuideMedSegWrapper(args)

    checkpoint = torch.load('/home/biiteam/Storage-4T/YLF/LanGuideMedSeg/languidemedseg/LanGuideMedSeg-MICCAI2023-main/save_model_new/medseg.ckpt',map_location='cpu')["state_dict"]
    model.load_state_dict(checkpoint,strict=True)

    # dataloader
    # ds_test = QaTa(csv_path=args.qata_test_csv_path,
    #                 root_path=args.qata_test_root_path,
    #                 tokenizer=args.bert_type,
    #                 image_size=args.image_size,
    #                 mode='test')
    
    
    # ds_test = MosMed(csv_path=args.mosmed_test_csv_path,
    #                 root_path=args.mosmed_test_root_path,
    #                 tokenizer=args.bert_type,
    #                 image_size=args.image_size,
    #                 mode='valid')
    ds_test = MosMed(csv_path=args.val_csv_path,
                    root_path=args.val_root_path,
                    tokenizer=args.bert_type,
                    image_size=args.image_size,
                    mode='valid')


    dl_test = DataLoader(ds_test, batch_size=args.valid_batch_size, shuffle=False, num_workers=8)

    trainer = pl.Trainer(accelerator='gpu',
                         devices=1,
                          deterministic=True
                          ) 
    model.eval()
    trainer.test(model, dl_test) 

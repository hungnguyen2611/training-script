import random
from argparse import ArgumentParser

import numpy as np
import pytorch_lightning as pl
import torch
from lit_data import LitDataModule
from lit_model import LitModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger


def set_random_seeds(random_seed=42):
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)




def main(args):
    
    data = LitDataModule(
        batch_size=args.batch_size, 
        data_dir=args.data_path,
        num_workers=4
    )
    
    data.setup("train")
    checkpoint_callback = ModelCheckpoint(
        save_top_k=2,
        monitor="val/loss",
        mode="min",
        dirpath=f"./checkpoint/{args.model}_weight_exp/",
        filename=f"{args.model}-{{epoch:02d}}-{{val/loss:.2f}}",
        save_weights_only=True
    )

    # wandb_logger = WandbLogger(project="proj_dummy")
    wandb_logger = WandbLogger(project="sns_classification")
    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[checkpoint_callback],
        devices=1,
        accelerator="gpu",
        max_epochs=args.epochs,
        logger=wandb_logger)

    train_dataloader = data.train_dataloader()
    valid_dataloader = data.val_dataloader()

    model = LitModule(args.model, lr=args.lr, progressive=True if args.progressive==1 else False, scheduler=True)
    
    trainer.fit(
        model=model, 
        train_dataloaders=train_dataloader, 
        val_dataloaders=valid_dataloader
    )



if __name__ == "__main__":
    set_random_seeds(42)
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default='convnext-pico')
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, help="Learning rate.", default=0.1)
    parser.add_argument("--progressive", type=int, default=0)
    pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    main(args)

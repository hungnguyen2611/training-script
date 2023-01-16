from argparse import ArgumentParser

import numpy as np
import pytorch_lightning as pl
from lit_data import LitDataModule
from lit_model import LitModule



def main(args):
    data = LitDataModule(
            batch_size=args.batch_size, 
            num_workers=4,
            data_dir=args.data_path
        )

    data.setup("test")

    trainer = pl.Trainer(
        devices=1,
        accelerator="cpu"
    )
    test_dataloader = data.test_dataloader()
    model = LitModule(net=args.model, lr=0.1)
    trainer.test(model=model, 
                dataloaders=test_dataloader
        )





if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default='convnext-small')
    # parser.add_argument("--weight_path", type=str)
    parser.add_argument("--batch_size", type=int, default=9)
    parser.add_argument("--data_path", type=str)
    args = parser.parse_args()
    main(args)
import glob
import os

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


def read_data_insta(label_file) -> pd.DataFrame:
    return pd.read_csv(label_file)

class InstaDataset(Dataset):
    def __init__(self, img_dir, label_file, transform=None, is_train=False):
        self.img_dir = img_dir
        self.df = read_data_insta(label_file)
        self.class_dict = {
            "p": 0,
            "o": 1,
            "pet": 2
        }
        p_df = self.df[self.df["label"]=="p"]
        pet_df = self.df[self.df["label"]=="pet"]
        o_df = self.df[self.df["label"]=="o"]
        if is_train:
            p_df = p_df.sample(n=4000)
            o_df = o_df.sample(n=4000)

        self.df = pd.concat([p_df, pet_df, o_df])
        self.transform = transform
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.img_dir, f"{self.df.iloc[idx, 0]}.jpg"))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = self.class_dict[self.df.iloc[idx, 1]]
        label = np.array(label, dtype=np.int_)
        return img, label



class InstaFolderDataset(Dataset):
    def __init__(self, img_dir, additional_dir=None, additional_dir2=None, transform=None, is_train=False, debug=False):
        self.img_dir = glob.glob(img_dir + '/*/*')
        if additional_dir:
            self.img_dir = self.img_dir + glob.glob(additional_dir+'/*/*')
            # self.img_dir = self.img_dir + glob.glob(additional_dir2+'/*/*')
        self.class_dict = {
            "p_imgs": 0,
            "org_imgs": 1,
            "pet_imgs": 2
        }
        self.transform = transform
        self.debug = debug
        
    def __len__(self):
        return len(self.img_dir)

    def __getitem__(self, idx):
        img = Image.open(self.img_dir[idx])
        class_name = os.path.basename(os.path.dirname(self.img_dir[idx]))
        img_name = os.path.basename(self.img_dir[idx])
        label = self.class_dict[class_name]
        if img.mode != 'RGB':
            img = img.convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = np.array(label, dtype=np.int_)
        if self.debug:
            return img, label, img_name
        return img, label


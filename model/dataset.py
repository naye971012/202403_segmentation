import torch
from torch.utils.data import Dataset
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from transformers import AutoImageProcessor
import numpy as np
import albumentations as A

from utils import get_img_762bands, get_mask_arr



def get_dataset(args):
    train_csv_path = os.path.join(args.data_path,"train_meta.csv")
    test_csv_path = os.path.join(args.data_path,"test_meta.csv")

    train_csv = pd.read_csv(train_csv_path)
    test_csv = pd.read_csv(test_csv_path)
    
    train_csv, valid_csv = train_test_split(train_csv, 
                                            test_size=args.validation_ratio, 
                                            random_state=args.seed)
    

    train_dataset = Traindataset(args=args,
                                 csv=train_csv,
                                 is_valid=False)
    valid_dataset = Traindataset(args=args,
                                 csv=valid_csv,
                                 is_valid=True)
    test_dataset = Testdataset(args=args,
                                 csv=test_csv_path)

    return (train_dataset, valid_dataset, test_dataset)


class Traindataset(Dataset):
    def __init__(self, args, csv, is_valid:bool):
        self.args = args
        self.csv = csv
        self.is_valid = is_valid
        
        self.transform = A.Compose([
            A.RandomRotate90(p=0.25),
            A.HorizontalFlip(p=0.25),
            A.VerticalFlip(p=0.25)
        ])
        
    def __len__(self):
        return len(self.csv)
    
    def __getitem__(self, index):
        image_path = self.csv.iloc[index]['train_img']
        image_path = os.path.join(os.path.join(self.args.data_path,"train_img"),image_path)
        
        mask_path = self.csv.iloc[index]['train_mask']
        mask_path = os.path.join(os.path.join(self.args.data_path,"train_mask"),mask_path)
        
        
        image_nparray = get_img_762bands(image_path) #[256,256,10] -> [256,256,10]
        mask_nparray = get_mask_arr(mask_path).reshape(256,256) # [256,256]
        
        if not self.is_valid:
            transformed = self.transform(image=image_nparray, mask = mask_nparray)
            image_nparray = transformed['image']
            mask_nparray = transformed['mask']
            # need to transform
        
        output = {
            "image": image_nparray,
            "annotation": mask_nparray
        }
        
        return output


class Testdataset(Dataset):
    def __init__(self, args, csv):
        self.args = args
        self.csv = csv

    def __len__(self):
        return len(self.csv)
    
    def __getitem__(self, index):
        image_path = self.csv.iloc[index]['test_img']
        image_path = os.path.join(os.path.join(self.args.data_path,"test_img"),image_path)
        
        image_nparray = get_img_762bands(image_path) #[256,256,10]
        
        output = {
            "image": image_nparray,
        }
            
        return output


import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class ICDARDataset(Dataset):

    def __init__(self, csv_filepath, root_dir, transforms=None, convert_rgb=True, mask_generator=None):
        self.transforms = transforms
        self.convert_rgb = convert_rgb
        self.mask_generator = mask_generator
        
        df = pd.read_csv(csv_filepath, sep=';')
        df['img_path'] = root_dir + os.sep + df['FILENAME']
        self.data = df.loc[df.img_path.map(os.path.exists)].reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        img_path = self.data.loc[idx, 'img_path']
        
        image = Image.open(img_path)
        
        if self.convert_rgb:
            image = image.convert('RGB')
        
        if self.transforms is not None:
            image = self.transforms(image)
        
        return image, self.data.loc[idx, 'SCRIPT_TYPE']

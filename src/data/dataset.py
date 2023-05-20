import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class ICDARDataset(Dataset):

    def __init__(self, csv_filepath, root_dir, transforms=None, convert_rgb=True, mask_generator=None):
        self.root_dir = root_dir
        self.transforms = transforms
        self.data = pd.read_csv(csv_filepath, sep=';')
        self.convert_rgb = convert_rgb
        self.mask_generator = mask_generator

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.root_dir, self.data.loc[idx, 'FILENAME'])
        
        try:
            image = Image.open(img_path)
        except Exception as ex:
            return None

        if self.convert_rgb:
            image = image.convert('RGB')
        
        if self.transforms is not None:
            image = self.transforms(image)
        
        return image, self.data.loc[idx, 'SCRIPT_TYPE']

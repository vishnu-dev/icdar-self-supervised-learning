import os
import torch
import pandas as pd
from skimage import io
from torch.utils.data import Dataset
import torchvision.transforms as T


class ICDARDataset(Dataset):
    
    def __init__(self, csv_filepath, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = pd.read_csv(csv_filepath, sep=';')

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_path = os.path.join(self.root_dir, self.data.loc[idx, 'FILE_NAME'])
        image = T.ToTensor()(io.imread(img_path))
        
        sample = {
            'image': image,
            'script_type': self.data.loc[idx, 'Script_type_ICDAR2017'],
            'script_date': self.data.loc[idx, 'DATE_ICDAR']
        }
        
        if self.transform is not None:
            sample['image'] = self.transform(sample['image'])
            sample['positive_pair'] = self.transform.__repr__()
        
        return sample

    
    
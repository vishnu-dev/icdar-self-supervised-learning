import os
import pandas as pd
import cv2
import numpy as np
import torch
from skimage import io
from torch.utils.data import Dataset
import os
from image_utils import show_image_list


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
        image = io.imread(img_path)
        
        sample = {
            'image': image,
            'script_type': self.data.loc[idx, 'Script_type_ICDAR2017'],
            'script_date': self.data.loc[idx, 'DATE_ICDAR']
        }
        
        if self.transform is not None:
            sample = self.transform(sample)
        
        return sample


def load_icdar_task1_task3(data_dir, label_file_path, num_samples=None, show_samples_num=1, show_width=300):
    label_df = pd.read_csv(label_file_path, sep=';', nrows=num_samples)
    data = label_df.to_dict(orient='records')
    data_available = []
    
    for file in data:
        image_path = os.path.join(data_dir, file['FILE_NAME'])
        if os.path.exists(image_path):
            im = cv2.imread(image_path)
            im = cv2.resize(im, (1000, 1000))
            file['im'] = np.asarray(im)
            data_available.append(file)
    
    show_image_list(
        list(map(lambda f: f.get('im'), data_available))[:show_samples_num],
        figsize=(20, 20),
        num_cols=3,
        grid=False
    )
    return data_available, label_df
    
    
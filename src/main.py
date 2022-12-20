from torch.utils.data import random_split, DataLoader
import pytorch_lightning as pl
from augment import PositivePairTransform
from dataset import ICDARDataset
import os
import torchvision.transforms as T
from image_utils import show_image_list
from tqdm import tqdm

from simclr.model import SimCLR


def train_test(root_dir, label_file_path):
    # data
    dataset = ICDARDataset(root_dir=root_dir, csv_filepath=label_file_path, transform=T.ToTensor())
    icdar_train, icdar_val = random_split(dataset, [1800, 200])
    
    train_loader = DataLoader(icdar_train, batch_size=32)
    val_loader = DataLoader(icdar_val, batch_size=32)
    
    # model
    model = SimCLR(batch_size=32, num_samples=10, max_epoch=10)
    
    # training
    trainer = pl.Trainer(accelerator='cpu', num_nodes=8, precision=16, limit_train_batches=0.5)
    trainer.fit(model, train_loader, val_loader)


def test_run():
    proj_dir = 'shared/FAU Study/Project/PR Project'
    
    data_dir = os.path.join(
        os.path.expanduser('~'),
        proj_dir,
        'Data/ICDAR2017_CLaMM_task1_task3/'
    )
    
    label_file_path = os.path.join(
        data_dir,
        '@ICDAR2017_CLaMM_task1_task3.csv'
    )
    
    train_test(data_dir, label_file_path)
    #
    # num_images_show = 10
    #
    # icdar_dataset = ICDARDataset(
    #     csv_filepath=label_file_path,
    #     root_dir=data_dir,
    #     transform=PositivePairTransform()
    # )
    #
    # images = []
    # labels = []
    #
    # for i in tqdm(range(num_images_show)):
    #     sample = icdar_dataset[i]
    #     std_img = T.RandomResizedCrop((512, 512))(sample['image'])
    #     images.append(std_img[0, :].numpy())
    #     labels.append(sample['positive_pair'])
    #
    # show_image_list(images, grid=False, num_cols=3, list_titles=labels)


if __name__ == '__main__':
    test_run()
    

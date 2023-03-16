import warnings
from pl_bolts.utils.stability import UnderReviewWarning
from pl_bolts.models.self_supervised import SimCLR
from torch.utils.data import random_split, DataLoader
from augment import PositivePairTransform
from pl_bolts.models.self_supervised.simclr.transforms import SimCLRTrainDataTransform, SimCLREvalDataTransform
from dataset import ICDARDataset
from evaluate import plot_features
from model import train_simclr
import os


def train_test(root_dir, is_eval=False):

    data_dir = os.path.join(root_dir, 'data/ICDAR2017_CLaMM_Training')
    label_file_path = os.path.join(data_dir, '@ICDAR2017_CLaMM_Training.csv')

    # data
    train_dataset = ICDARDataset(label_file_path, data_dir, transforms=SimCLRTrainDataTransform())
    val_dataset = ICDARDataset(label_file_path, data_dir, transforms=SimCLREvalDataTransform())

    if not is_eval:
        model = train_simclr(
            root_dir,
            train_dataset,
            val_dataset,
            batch_size=32,
            max_epochs=1,
            gpus=-1,
            num_samples=len(train_dataset),
            dataset='icdar'
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
        num_workers=os.cpu_count()
    )

    checkpoint = os.path.join(
        root_dir,
        'SimCLR/lightning_logs/version_6/checkpoints/epoch=99-step=11000.ckpt'
    )

    simclr_trained = SimCLR.load_from_checkpoint(checkpoint)
    plot_features(simclr_trained, val_loader, 2048, 32, 1000)


def test_run():

    proj_dir = os.path.join(
        os.path.expanduser('~'),
        'icdar'
    )
    train_test(proj_dir, is_eval=False)


if __name__ == '__main__':
    test_run()
    

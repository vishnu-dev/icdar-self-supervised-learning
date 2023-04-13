import os
import pytorch_lightning as pl
from pl_bolts.models.self_supervised import BYOL
from pl_bolts.models.self_supervised.simclr import SimCLRTrainDataTransform, SimCLREvalDataTransform
from torch.utils.data import DataLoader
from data.dataset import ICDARDataset


def train_byol(proj_dir, train_data, val_data, batch_size, max_epochs=500, num_workers=os.cpu_count(), **kwargs):

    trainer = pl.Trainer(
        default_root_dir=os.path.join(proj_dir, 'BYOL'),
        accelerator='gpu',
        devices=-1,
        max_epochs=max_epochs,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                save_weights_only=True, mode='min', monitor='val_loss'
            )
        ],
        enable_progress_bar=True,
        precision=16
    )

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(proj_dir, 'BYOL.ckpt')
    if os.path.isfile(pretrained_filename):
        print(f'Found pretrained model at {pretrained_filename}, loading...')
        # Automatically loads the model with the saved hyperparameters
        byol = BYOL.load_from_checkpoint(pretrained_filename)
    else:
        train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            num_workers=num_workers
        )
        val_loader = DataLoader(
            val_data,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            num_workers=num_workers
        )

        byol = BYOL(max_epochs=max_epochs, batch_size=batch_size, **kwargs)
        byol.cuda()
        trainer.fit(byol, train_loader, val_loader)

        # Load the best checkpoint after training
        byol = BYOL.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    return byol


def train_test(root_dir, is_eval=False):
    data_dir = os.path.join(root_dir, 'data/ICDAR2017_CLaMM_Training')
    label_file_path = os.path.join(data_dir, '@ICDAR2017_CLaMM_Training.csv')

    # data
    train_dataset = ICDARDataset(label_file_path, data_dir, transforms=SimCLRTrainDataTransform())
    val_dataset = ICDARDataset(label_file_path, data_dir, transforms=SimCLREvalDataTransform())

    if not is_eval:
        model = train_byol(
            root_dir,
            train_dataset,
            val_dataset,
            batch_size=64,
            max_epochs=100,
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
        'BYOL/lightning_logs/version_6/checkpoints/epoch=99-step=11000.ckpt'
    )

    byol_trained = BYOL.load_from_checkpoint(checkpoint)
    # plot_features(byol_trained, val_loader, 2048, 32, 1000)


def test_run():
    proj_dir = os.path.join(
        os.path.expanduser('~'),
        'icdar'
    )
    train_test(proj_dir, is_eval=False)


if __name__ == '__main__':
    test_run()


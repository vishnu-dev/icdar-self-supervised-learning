import torch
from pl_bolts.models.self_supervised import SimCLR
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import os


def train_simclr(proj_dir, train_data, val_data, batch_size, max_epochs=500, num_workers=os.cpu_count(), **kwargs):

    trainer = pl.Trainer(
        default_root_dir=os.path.join(proj_dir, 'SimCLR'),
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
    pretrained_filename = os.path.join(proj_dir, 'SimCLR.ckpt')
    if os.path.isfile(pretrained_filename):
        print(f'Found pretrained model at {pretrained_filename}, loading...')
        # Automatically loads the model with the saved hyperparameters
        simclr = SimCLR.load_from_checkpoint(pretrained_filename)
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

        simclr = SimCLR(max_epochs=max_epochs, batch_size=batch_size, **kwargs)
        simclr.cuda()
        trainer.fit(simclr, train_loader, val_loader)

        # Load the best checkpoint after training
        simclr = SimCLR.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    return simclr

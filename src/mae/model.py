from src.image_utils import show_image_list
from src.mae.vit import ViTBlocks
from torchvision.models import vit_b_32
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from src.dataset import ICDARDataset
from typing import Tuple, Union
import pytorch_lightning as pl
import torch
import os
from lightly.data.collate import MAECollateFunction
from lightly.models import utils
from lightly.models.modules import masked_autoencoder

    
class MAE(pl.LightningModule):
    def __init__(self):
        super().__init__()

        decoder_dim = 512
        vit = vit_b_32(pretrained=False)
        self.mask_ratio = 0.75
        self.patch_size = vit.patch_size
        self.sequence_length = vit.seq_length
        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.backbone = masked_autoencoder.MAEBackbone.from_vit(vit)
        self.decoder = masked_autoencoder.MAEDecoder(
            seq_length=vit.seq_length,
            num_layers=1,
            num_heads=16,
            embed_input_dim=vit.hidden_dim,
            hidden_dim=decoder_dim,
            mlp_dim=decoder_dim * 4,
            out_dim=vit.patch_size**2 * 3,
            dropout=0,
            attention_dropout=0,
        )
        self.criterion = torch.nn.MSELoss()

    def forward_encoder(self, images, idx_keep=None):
        return self.backbone.encode(images, idx_keep)

    def forward_decoder(self, x_encoded, idx_keep, idx_mask):
        # build decoder input
        batch_size = x_encoded.shape[0]
        x_decode = self.decoder.embed(x_encoded)
        x_masked = utils.repeat_token(
            self.mask_token, (batch_size, self.sequence_length)
        )
        x_masked = utils.set_at_index(x_masked, idx_keep, x_decode.type_as(x_masked))

        # decoder forward pass
        x_decoded = self.decoder.decode(x_masked)

        # predict pixel values for masked tokens
        x_pred = utils.get_at_index(x_decoded, idx_mask)
        x_pred = self.decoder.predict(x_pred)
        return x_pred

    def training_step(self, batch, batch_idx):
        images, _ = batch

        batch_size = images.shape[0]
        idx_keep, idx_mask = utils.random_token_mask(
            size=(batch_size, self.sequence_length),
            mask_ratio=self.mask_ratio,
            device=images.device,
        )
        x_encoded = self.forward_encoder(images, idx_keep)
        x_pred = self.forward_decoder(x_encoded, idx_keep, idx_mask)

        # get image patches for masked tokens
        patches = utils.patchify(images, self.patch_size)
        # must adjust idx_mask for missing class token
        target = utils.get_at_index(patches, idx_mask - 1)

        loss = self.criterion(x_pred, target)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.parameters(), lr=1.5e-4)
        return optim

    def predict(self, images):
        x_encoded = self.forward_encoder(images)
        x_pred = self.forward_decoder(x_encoded)
        x_pred = x_pred[:, 1:]  # drop class token
        return x_pred


def run(root_dir, train=False, test=False):
    max_epochs = 50
    batch_size = 32
    image_width = 64
    num_workers = os.cpu_count()
    
    data_dir = os.path.join(root_dir, 'data/ICDAR2017_CLaMM_Training')
    label_file_path = os.path.join(data_dir, '@ICDAR2017_CLaMM_Training.csv')
    
    tform = transforms.Compose([
        transforms.RandomResizedCrop(size=image_width),
        transforms.ToTensor()
    ])

    icdar_dataset = ICDARDataset(label_file_path, data_dir, transforms=tform, convert_rgb=True)

    total_count = len(icdar_dataset)
    train_count = int(0.7 * total_count)
    valid_count = int(0.2 * total_count)
    test_count = total_count - train_count - valid_count

    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        icdar_dataset,
        (train_count, valid_count, test_count)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=num_workers
    )
    val_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=num_workers
    )

    trainer = pl.Trainer(
        default_root_dir=os.path.join(root_dir, 'MAE'),
        accelerator='gpu',
        devices=-1,
        max_epochs=max_epochs,
        enable_progress_bar=True,
        precision=16
    )

    mae = MAE()
    mae.cuda()

    if train:
        trainer.fit(mae, train_loader)

    if test:
        # checkpoint = os.path.join(
        #     root_dir,
        #     f'MAE/lightning_logs/version_{version}/checkpoints/epoch={epoch_num}-step={step_num}.ckpt'
        # )
        trainer.test(mae, test_loader, ckpt_path="best")
        mae_trained = trainer.model
        mae_trained.eval()

        for test_images, test_labels in test_loader:
            print(test_images[0].shape)
            with torch.no_grad():
                out = mae_trained.predict(test_images)
            print(test_images[0].shape, out[0].shape)
            show_image_list([
                test_images[0].permute(1, 2, 0).cpu().detach().numpy(),
                out[0].cpu().detach().numpy()
            ])
            break


if __name__ == '__main__':
    proj_dir = os.path.join(
        os.path.expanduser('~'),
        'icdar'
    )
    # run(proj_dir, train=True)
    run(proj_dir, test=True)

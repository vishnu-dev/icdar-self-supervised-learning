from src.image_utils import show_image_list
from src.mae.vit import ViTBlocks
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from src.dataset import ICDARDataset
from typing import Tuple, Union
import pytorch_lightning as pl
import torch
import os

    
class MaskedAutoencoder(torch.nn.Module):
    """Masked Autoencoder for visual representation learning.
    Args:
        image_size: (height, width) of the input images.
        patch_size: side length of a patch.
        keep: percentage of tokens to process in the encoder. (1 - keep) is the percentage of masked tokens.
        enc_width: width (feature dimension) of the encoder.
        dec_width: width (feature dimension) of the decoder. If a float, it is interpreted as a percentage
            of enc_width.
        enc_depth: depth (number of blocks) of the encoder
        dec_depth: depth (number of blocks) of the decoder
    """
    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        patch_size: int = 16,
        keep: float = 0.25,
        enc_width: int = 768,
        dec_width: Union[int, float] = 0.25,
        enc_depth: int = 12,
        dec_depth: int = 8,
        channels: int = 3
    ):
        super().__init__()

        assert image_size[0] % patch_size == 0 and image_size[1] % patch_size == 0

        self.image_size = image_size
        self.channels = channels
        self.patch_size = patch_size
        self.keep = keep
        self.n = (image_size[0] * image_size[1]) // patch_size**2  # number of patches

        if isinstance(dec_width, float) and dec_width > 0 and dec_width < 1:
            dec_width = int(dec_width * enc_width)
        else:
            dec_width = int(dec_width)
        self.enc_width = enc_width
        self.dec_width = dec_width
        self.enc_depth = enc_depth
        self.dec_depth = dec_depth

        # linear patch embedding
        self.embed_conv = torch.nn.Conv2d(self.channels, enc_width, patch_size, patch_size)

        # mask token and position encoding
        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, dec_width, requires_grad=True))
        self.register_buffer('pos_encoder', self.pos_encoding(self.n, enc_width).requires_grad_(False))
        self.register_buffer('pos_decoder', self.pos_encoding(self.n, dec_width).requires_grad_(False))

        # encoder
        self.encoder = ViTBlocks(width=enc_width, depth=enc_depth, channels=channels)

        # linear projection from enc_width to dec_width
        self.project = torch.nn.Linear(enc_width, dec_width)

        # decoder
        self.decoder = ViTBlocks(width=dec_width, depth=dec_depth, end_norm=False, channels=channels)

        # linear projection to pixel dimensions
        self.pixel_project = torch.nn.Linear(dec_width, self.channels * patch_size**2)

        self.freeze_mask = False  # set to True to reuse the same mask multiple times

    @property
    def freeze_mask(self):
        """When True, the previously computed mask will be used on new inputs, instead of creating a new one."""
        return self._freeze_mask

    @freeze_mask.setter
    def freeze_mask(self, val: bool):
        self._freeze_mask = val

    @staticmethod
    def pos_encoding(n: int, d: int, k: int = 10000):
        """Create sine-cosine positional embeddings.
        Args:
            n: the number of embedding vectors, corresponding to the number of tokens (patches) in the image.
            d: the dimension of the embeddings
            k: value that determines the maximum frequency (10,000 by default)

        Returns:
            (n, d) tensor of position encoding vectors
        """
        x = torch.meshgrid(
            torch.arange(n, dtype=torch.float32),
            torch.arange(d, dtype=torch.float32),
            indexing='ij'
        )
        pos = torch.zeros_like(x[0])
        pos[:, ::2] = x[0][:, ::2].div(torch.pow(k, x[1][:, ::2].div(d // 2))).sin_()
        pos[:, 1::2] = x[0][:,1::2].div(torch.pow(k, x[1][:,1::2].div(d // 2))).cos_()
        return pos

    @staticmethod
    def generate_mask_index(bs: int, n_tok: int, device: str='cpu'):
        """Create a randomly permuted token-index tensor for determining which tokens to mask.
        Args:
            bs: batch size
            n_tok: number of tokens per image
            device: the device where the tensors should be created
        Returns:
            (bs, 1) tensor of batch indices [0, 1, ..., bs - 1]^T
            (bs, n_tok) tensor of token indices, randomly permuted
        """
        bidx = torch.arange(bs, device=device)[..., None] 
        idx = torch.stack([torch.randperm(n_tok, device=device) for _ in range(bs)], 0)
        return bidx, idx

    def image_as_tokens(self, x: torch.Tensor):
        """Reshape an image of shape (b, c, h, w) to a set of vectorized patches
        of shape (b, h*w/p^2, c*p^2). In other words, the set of non-overlapping
        patches of size (3, p, p) in the image are turned into vectors (tokens);
        dimension 1 of the output indexes each patch.
        """
        b, c, h, w = x.shape
        p = self.patch_size
        x = x.reshape(b, c, h // p, p, w // p, p).permute(0, 2, 4, 1, 3, 5)
        x = x.reshape(b, (h * w) // p**2, c * p * p)
        return x

    def tokens_as_image(self, x: torch.Tensor):
        """Reshape a set of token vectors into an image. This is the reverse operation
        of `image_as_tokens`.
        """
        b = x.shape[0]
        im, p = self.image_size, self.patch_size
        hh, ww = im[0] // p, im[1] // p
        x = x.reshape(b, hh, ww, self.channels, p, p).permute(0, 3, 1, 4, 2, 5)
        x = x.reshape(b, self.channels, p * hh, p * ww)
        return x

    def masked_image(self, x: torch.Tensor):
        """Return a copy of the image batch, with the masked patches set to 0. Used
        for visualization.
        """
        x = self.image_as_tokens(x).clone()
        bidx = torch.arange(x.shape[0], device=x.device)[:,None]
        x[bidx, self.idx[:, int(self.keep * self.n):]] = 0
        return self.tokens_as_image(x)

    def embed(self, x: torch.Tensor):
        return self.embed_conv(x).flatten(2).transpose(1, 2)

    def mask_input(self, x: torch.Tensor):
        """Mask the image patches uniformly at random, as described in the paper: the patch tokens are
        randomly permuted (per image), and the first N are returned, where N corresponds to percentage
        of patches kept (not masked).

        Returns the masked (truncated) tokens. The mask indices are saved as `self.bidx` and `self.idx`.
        """
        # create a new mask if self.freeze_mask is False, or if no mask has been created yet
        if not hasattr(self, 'idx') or not self.freeze_mask:
            self.bidx, self.idx = self.generate_mask_index(x.shape[0], x.shape[1], x.device)

        k = int(self.keep * self.n)
        x = x[self.bidx, self.idx[:, :k]]
        return x

    def forward_features(self, x: torch.Tensor):
        x = self.embed(x)
        x = x + self.pos_encoder
        x = self.mask_input(x)
        x = self.encoder(x)

        return x

    def forward(self, x: torch.Tensor):
        x = self.forward_features(x)
        x = self.project(x)

        k = self.n - x.shape[1]  # number of masked tokens
        mask_toks = self.mask_token.repeat(x.shape[0], k, 1)
        x = torch.cat([x, mask_toks], 1)
        x = x[self.bidx, self.idx.argsort(1)] + self.pos_decoder
        x = self.decoder(x)
        x = self.pixel_project(x)

        return x


class MAE(pl.LightningModule):
    """Masked Autoencoder LightningModule.
    Args:
        image_size: (height, width) of the input images.
        patch_size: size of the image patches.
        keep: percentage of tokens to keep. (1 - keep) is the percentage of masked tokens.
        enc_width: width of the encoder features.
        dec_width: width of the decoder features.
        enc_depth: depth of the encoder.
        dec_depth: depth of the decoder.
        lr: learning rate
        train_steps: number of training steps.
    """
    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        channels: int = 3,
        patch_size: int = 16,
        keep: float = 0.25,
        enc_width: int = 768,
        dec_width: float = 0.25,
        enc_depth: int = 12,
        dec_depth: int = 4,
        lr: float = 1.5e-4,
        train_steps: int = 100000,
    ):
        super().__init__()

        self.mae = MaskedAutoencoder(
            image_size=image_size,
            channels=channels,
            patch_size=patch_size,
            keep=keep,
            enc_width=enc_width,
            enc_depth=enc_depth,
            dec_width=dec_width,
            dec_depth=dec_depth,
        )

        self.keep = keep
        self.n = self.mae.n
        self.lr = lr
        self.train_steps = train_steps

        self.save_hyperparameters()

    def training_step(self, batch: Union[tuple, list]):
        x, _ = batch
        pred = self.mae(x)
        loss = self.masked_mse_loss(x, pred)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch: Union[tuple, list], batch_idx: int):
        x, _ = batch
        pred = self.mae(x)
        loss = self.masked_mse_loss(x, pred)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.parameters(), lr=self.lr, betas=(.9, .95), weight_decay=0.05)
        schedule = torch.optim.lr_scheduler.OneCycleLR(
            optim,
            max_lr=self.lr,
            total_steps=self.train_steps,
            pct_start=0.1,
            cycle_momentum=False,
        )
        return {
            'optimizer': optim, 
            'lr_scheduler': {'scheduler': schedule, 'interval': 'step'}
        }

    def masked_mse_loss(self, img: torch.Tensor, recon: torch.Tensor):
        # turn the image into patch-vectors for comparison to model output
        x = self.mae.image_as_tokens(img)
        bidx = torch.arange(x.shape[0], device=x.device)[:, None]  # B x 1
        # only compute on the mask token outputs, which is everything after the first (n * keep)
        idx = self.mae.idx[:, int(self.keep * self.n):]
        x = x[bidx, idx]
        y = recon[bidx, idx]
        return torch.nn.functional.mse_loss(x, y)


def run(root_dir, train=False, test=False):
    max_epochs = 10
    batch_size = 8
    image_width = 224
    num_workers = os.cpu_count()
    
    data_dir = os.path.join(root_dir, 'data/ICDAR2017_CLaMM_Training')
    label_file_path = os.path.join(data_dir, '@ICDAR2017_CLaMM_Training.csv')
    
    tform = transforms.Compose([
        transforms.RandomResizedCrop(size=image_width),
        transforms.ToTensor()
    ])

    icdar_dataset = ICDARDataset(label_file_path, data_dir, transforms=tform, convert_rgb=False)

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

    if train:
        mae = MAE(image_size=(image_width, image_width), channels=1)
        mae.cuda()
        trainer = pl.Trainer(
            default_root_dir=os.path.join(root_dir, 'MAE'),
            accelerator='gpu',
            devices=-1,
            max_epochs=max_epochs,
            strategy='ddp',
            callbacks=[
                pl.callbacks.ModelCheckpoint(
                    save_weights_only=True, every_n_epochs=1, save_top_k=-1, save_on_train_epoch_end=True
                )
            ],
            enable_progress_bar=True,
            precision=16
        )
        trainer.fit(mae, train_loader)

    if test:
        checkpoint = os.path.join(
            root_dir,
            'MAE/lightning_logs/version_0/checkpoints/epoch=7-step=2472.ckpt'
        )
        mae_trained = MAE.load_from_checkpoint(checkpoint)
        mae_trained.eval()

        for test_images, test_labels in test_loader:
            print(test_images[0].shape)
            with torch.no_grad():
                out = mae_trained(test_images)
                out = mae_trained.mae.tokens_as_image(out)
            print(test_images[0].shape, out[0].shape)
            show_image_list([
                test_images[0].permute(1, 2, 0).cpu().detach().numpy(),
                out[0].permute(1, 2, 0).cpu().detach().numpy()
            ])
            break


if __name__ == '__main__':
    proj_dir = os.path.join(
        os.path.expanduser('~'),
        'icdar'
    )
    run(proj_dir, train=True)
    # run(proj_dir, test=True)

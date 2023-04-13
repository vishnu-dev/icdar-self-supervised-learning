import torch
from torch.nn import functional as F
import pytorch_lightning as pl

from models.simclr.encoder import ResNet50Encoder
from models.simclr.head import ProjectionHead


class SimCLR(pl.LightningModule):
    def __init__(self, batch_size, num_samples, max_epoch):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = ResNet50Encoder()
        self.projection = ProjectionHead(1000)
    
    def forward(self, x):
        embedding = self.encoder(x)
        return embedding
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.projection(z)
        loss = F.mse_loss(x_hat, x)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.projection(z)
        loss = F.mse_loss(x_hat, x)
        self.log('val_loss', loss)


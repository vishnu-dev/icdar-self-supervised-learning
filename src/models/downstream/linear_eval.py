from torch import no_grad
from torch.optim import Adam
from torch.nn import Linear, CrossEntropyLoss
from torch.nn.functional import sigmoid
import torchmetrics
import pytorch_lightning as pl


class DownstreamClassifier(pl.LightningModule):
    
    def __init__(self, base_model=None, features=2048, num_classes=13, learning_rate=1e-2, batch_size=64):
        super().__init__()
        
        self.save_hyperparameters()
                
        self.learning_rate = learning_rate
        
        self.num_classes = num_classes
        
        self.base_model = base_model

        self.classifier = Linear(features, num_classes)
        
        self.criterion = CrossEntropyLoss()
        
    def forward(self, x):
        
        with no_grad():
            x = self.base_model(x)
        x = self.classifier(x)
                
        return x

    def training_step(self, batch, batch_idx):
        (x1, x2, _), label = batch
        
        y_hat = self(x1)
        loss = self.criterion(y_hat, label)
        self.log('train_loss', loss)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        (x1, x2, _), label = batch
        
        y_hat = self(x1)
        loss = self.criterion(y_hat, label)
        self.log('val_loss', loss)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        (x1, x2, _), label = batch
        
        y_hat = self(x1)
        acc_metric = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes).to(self.device)
        acc = acc_metric(y_hat, label)
        self.log('test_acc', acc, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.learning_rate)

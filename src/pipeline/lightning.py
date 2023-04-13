import os

import pytorch_lightning as pl


class LightningPipeline:

    def __init__(self, root_dir, model_class, mode, data_loader, max_epochs, batch_size):

        pl.seed_everything(42)

        self.model_class = model_class
        self.mode = mode
        self.data_loader = data_loader
        self.root_dir = root_dir
        self.max_epochs = max_epochs
        self.batch_size = batch_size

    def init_model(self):
        return self.model_class

    def init_trainer(self, model):
        return pl.Trainer(
            default_root_dir=os.path.join(self.root_dir, 'trained_models', model.__class__.__name__),
            accelerator='gpu',
            devices=-1,
            max_epochs=self.max_epochs,
            callbacks=[
                pl.callbacks.ModelCheckpoint(mode='min', monitor='val_loss')
            ],
            enable_progress_bar=True,
            precision=16
        )

    def run(self):

        model = self.init_model()
        trainer = self.init_trainer(model)

        if self.mode == 'train':
            trainer.fit(model, self.data_loader.get(self.mode), self.data_loader.get('val'))
        elif self.mode == 'test':
            trainer.test(model, self.data_loader.get(self.mode))
        else:
            raise ValueError(f'Invalid mode {self.mode}')

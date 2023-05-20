import os
import pytorch_lightning as pl
from pipeline.callback_factory import callback_factory


class LightningPipeline:
    def __init__(
        self, root_dir, model_class, mode, data_loader, max_epochs, batch_size
    ):
        pl.seed_everything(42)

        self.model_class = model_class
        self.mode = mode
        self.data_loader = data_loader
        self.root_dir = root_dir
        self.max_epochs = max_epochs
        self.batch_size = batch_size

        self.trainer = None

    def init_model(self):
        return self.model_class

    def init_trainer(self, model):
        model_save_dir = os.path.abspath(
                os.path.join(
                    self.root_dir, os.pardir, "trained_models", model.__class__.__name__
                )
            )
        print('Saving models to: ', model_save_dir)
        print('Callbacks: ', callback_factory(model.__class__.__name__))
        self.trainer = pl.Trainer(
            default_root_dir=model_save_dir,
            accelerator="gpu",
            devices=-1,
            max_epochs=self.max_epochs,
            callbacks=callback_factory(model.__class__.__name__),
            enable_progress_bar=True,
            precision=16,
            log_every_n_steps=20
        )

    def run(self):
        model = self.init_model()
        self.init_trainer(model)

        if self.mode == "train":
            self.trainer.fit(
                model, self.data_loader.get("train"), self.data_loader.get("val")
            )
        elif self.mode == "test":
            self.trainer.test(model, self.data_loader.get(self.mode))
        else:
            raise ValueError(f"Invalid mode {self.mode}")

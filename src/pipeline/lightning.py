import os
import pytorch_lightning as pl
from src.pipeline.callback_factory import callback_factory


class LightningPipeline:
    def __init__(
        self, root_dir, model_class, mode, data_loader, batch_size, trainer_cfg
    ):
        # pl.seed_everything(42, workers=True)

        self.model_class = model_class
        self.mode = mode
        self.data_loader = data_loader
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.trainer_cfg = trainer_cfg

        self.trainer = None

    def init_model(self):
        return self.model_class

    def init_trainer(self, model):
        self.trainer = pl.Trainer(
            default_root_dir=self.root_dir,
            accelerator=self.trainer_cfg.accelerator,
            strategy=self.trainer_cfg.strategy or None,
            devices=self.trainer_cfg.devices,
            max_epochs=self.trainer_cfg.max_epochs,
            callbacks=callback_factory(model.__class__.__name__),
            enable_progress_bar=True,
            precision=self.trainer_cfg.precision,
            log_every_n_steps=self.trainer_cfg.log_every_n_steps,
            accumulate_grad_batches=self.trainer_cfg.get('accumulate_grad_batches', 1),
            profiler='simple',
            # benchmark=True,
            # deterministic=True
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

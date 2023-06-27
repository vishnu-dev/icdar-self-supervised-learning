import os
import hydra
from glob import glob
from data.data_factory import data_factory
from data.transforms import transform_factory
from data.collate import collate_factory
from pipeline.lightning import LightningPipeline
from models.backbone_factory import backbone_factory
from models.model_factory import model_factory
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path='config', config_name='config')
def execute(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    
    transforms = transform_factory(cfg.model.name, cfg.mode.name)

    collate_fn = collate_factory(cfg.model.name)

    dataloaders = data_factory(
        cfg.dataset.name,
        cfg.dataset.root_dir,
        cfg.dataset.label_path,
        cfg.dataset.train_val_test_ratio,
        transforms,
        cfg.mode.name,
        cfg.dataset.batch_size,
        collate_fn,
        cfg.dataset.num_workers
    )
    
    for name, dataloader in dataloaders.items():
        print(f'{name} dataloader: ', len(dataloader.dataset))

    base_checkpoint_path = os.path.join(
        cfg.trainer.default_root_dir,
        cfg.model.base_model.classname,
        'lightning_logs',
        f'version_{cfg.model.base_model.checkpoint}',
        'checkpoints',
        '*.ckpt'
    )
    base_checkpoint = sorted(glob(base_checkpoint_path), reverse=True)[-1]

    base_model = backbone_factory(
        cfg.model.base_model.name,
        base_checkpoint,
        gpus=-1,
        max_epochs=cfg.trainer.max_epochs,
        batch_size=cfg.dataset.batch_size,
        dataset=cfg.dataset.name
    )

    model_class = model_factory(
        cfg.model.name,
        base_model=base_model,
        num_samples=len(dataloaders.get(cfg.mode.name).dataset),
        max_epochs=cfg.trainer.max_epochs,
        batch_size=cfg.dataset.batch_size,
        dataset=cfg.dataset.name,
        **cfg.model.params
    )

    pipeline = LightningPipeline(
        os.path.join(cfg.trainer.default_root_dir, model_class.__class__.__name__),
        model_class,
        cfg.mode.name,
        dataloaders,
        cfg.dataset.batch_size,
        cfg.trainer
    )
    pipeline.run()


if __name__ == "__main__":
    execute()

import os
from omegaconf import DictConfig, OmegaConf
import hydra
import json

from src.data.data_factory import data_factory
from src.data.transforms import transform_factory
from src.data.collate import collate_factory
from src.models.model_factory import model_factory
from src.pipeline.lightning import LightningPipeline


@hydra.main(version_base=None, config_path='config', config_name='config')
def execute(cfg: DictConfig):
    """
    Configuration based model training entry point.
    CLI arguments are passed as configuration overrides.
    
    Configurations are defined in the config directory.
    
    Args:
        cfg: The configuration object from hydra.
    
    Examples:
        >>> python train.py +experiment=simclr_bolts model.params.batch_size=128
        >>> python train.py +experiment=byol_paper
    
    """
    
    print(OmegaConf.to_yaml(cfg))
    cfg = hydra.utils.instantiate(cfg)
        
    transforms = transform_factory(
        cfg.model.name,
        cfg.mode.name,
        cfg.model.augmentations
    )
    
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
    
    # Prints the dataset size for each type of dataloader.
    for name, dataloader in dataloaders.items():
        print(f'{name} dataloader: ', len(dataloader.dataset))
    
    model_class = model_factory(
        cfg.model.name,
        num_samples=len(dataloaders.get(cfg.mode.name).dataset),
        max_epochs=cfg.trainer.max_epochs,
        batch_size=cfg.dataset.batch_size,
        dataset=cfg.dataset.name,
        learning_rate=cfg.model.learning_rate,
        **cfg.model.params
    )
    print('Model class: ', model_class.__class__.__name__)

    pipeline = LightningPipeline(
        os.path.join(cfg.trainer.default_root_dir, model_class.__class__.__name__),
        model_class,
        cfg.mode.name,
        dataloaders,
        cfg.dataset.batch_size,
        cfg.trainer
    )

    pipeline.run()


if __name__ == '__main__':
    execute()

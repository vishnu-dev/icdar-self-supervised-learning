import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path='../config', config_name='config')
def execute(cfg: DictConfig):
    """Testing the configuration management.

    Args:
        cfg (DictConfig): Hydra configuration object.
    
    Examples:
        >>> python config.py model.name=SimCLR model.params.batch_size=128
        >>> python config.py +experiment=simclr_bolts
    
    """
    
    cfg = hydra.utils.instantiate(cfg)
    
    print(OmegaConf.to_yaml(cfg))


if __name__ == '__main__':
    execute()
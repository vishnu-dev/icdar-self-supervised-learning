import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path='../config', config_name='config')
def execute(cfg: DictConfig):
    
    cfg = hydra.utils.instantiate(cfg)
    
    print(OmegaConf.to_yaml(cfg))


if __name__ == '__main__':
    execute()
import hydra
from pl_bolts.models.self_supervised.simclr import SimCLREvalDataTransform
from torchvision.transforms import Compose, ToTensor, Normalize
from lightly.transforms.mae_transform import MAETransform
from data.augment import NegativePairTransform
import torchvision.transforms as T


def transform_factory(model_name, mode, config, data_mean, data_std):

    if config is not None:
        augmentations = T.Compose(config.get('train'))
            
    transforms_dict = {
        'simclr': {
            'train': NegativePairTransform(transforms=augmentations),
            'val': SimCLREvalDataTransform(),
            'test': SimCLREvalDataTransform()
        },
        'mae': {
            # Defined scale as in the paper
            'train': MAETransform(min_scale=0.2, normalize=None),
            'val': Compose([ToTensor(), Normalize(data_mean, data_std)]),
            'test': Compose([ToTensor(), Normalize(data_mean, data_std)])
        },
        'byol': {
            'train': NegativePairTransform(transforms=augmentations),
            'val': SimCLREvalDataTransform(),
            'test': SimCLREvalDataTransform()
        },
        'downstream_linear': {
            'train': SimCLREvalDataTransform(),
            'val': SimCLREvalDataTransform(),
            'test': SimCLREvalDataTransform()
        }
    }
    try:
        return transforms_dict.get(model_name).get(mode)
    except KeyError:
        raise NotImplementedError(f'{model_name} {mode} transform not implemented')

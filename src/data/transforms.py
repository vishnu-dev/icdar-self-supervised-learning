from pl_bolts.models.self_supervised.simclr import SimCLREvalDataTransform
from torchvision.transforms import Compose, ToTensor
from lightly.transforms.mae_transform import MAETransform
from src.data.augment import NegativePairTransform
import torchvision.transforms as T


def transform_factory(model_name, mode, config=None):

    if config is not None and mode == 'train':
        augmentations = T.Compose(config.get('train'))
        online_augmentations = T.Compose(config.get('online'))
    else:
        augmentations = None
        online_augmentations = None
            
    transforms_dict = {
        'simclr': {
            'train': NegativePairTransform(transforms=augmentations, online_transforms=online_augmentations),
            'val': SimCLREvalDataTransform(),
            'test': SimCLREvalDataTransform()
        },
        'mae': {
            # Defined scale as in the paper
            'train': MAETransform(min_scale=0.2, normalize=None),
            'val': Compose([ToTensor()]),
            'test': Compose([ToTensor()])
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

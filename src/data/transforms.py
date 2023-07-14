from pl_bolts.models.self_supervised.simclr import SimCLREvalDataTransform
from torchvision.transforms import Compose, ToTensor
from lightly.transforms.mae_transform import MAETransform
from src.data.augment import PairTransform
import torchvision.transforms as T


def transform_factory(model_name, mode, config=None):
    """Transform factory for self-supervised models

    Args:
        model_name (str): Name of the model
        mode (str): Execution mode (train, test)
        config (dict, optional): Configuration parameters from hydra. Defaults to None.

    Raises:
        NotImplementedError: If transform is not implemented for the given model or mode

    Returns:
        torchvision.transforms.Compose: Transforms
    """

    if config is not None and mode == 'train':
        augmentations = T.Compose(config.get('train'))
        online_augmentations = T.Compose(config.get('online'))
    else:
        augmentations = None
        online_augmentations = None
            
    transforms_dict = {
        'simclr': {
            'train': PairTransform(transforms=augmentations, online_transforms=online_augmentations),
            'val': SimCLREvalDataTransform(),
            'test': SimCLREvalDataTransform()
        },
        'mae': {
            'train': augmentations,
            'val': Compose([
                T.Resize(256, interpolation=3),
                T.CenterCrop(224),
                T.ToTensor()
            ]),
            'test': Compose([
                T.Resize(256, interpolation=3),
                T.CenterCrop(224),
                T.ToTensor(),
            ])
        },
        'byol': {
            'train': PairTransform(transforms=augmentations, online_transforms=online_augmentations),
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
        return transforms_dict[model_name][mode]
    except KeyError:
        raise NotImplementedError(f'{model_name} {mode} transform not implemented')

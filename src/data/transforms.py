from pl_bolts.models.self_supervised.simclr import SimCLRTrainDataTransform, SimCLREvalDataTransform
from torchvision.transforms import Compose, ToTensor, Normalize
from lightly.transforms.mae_transform import MAETransform

transforms_dict = {
    'simclr': {
        'train': SimCLRTrainDataTransform(),
        'val': SimCLREvalDataTransform(),
        'test': SimCLREvalDataTransform()
    },
    'mae': {
        'train': MAETransform(),
        'val': Compose([ToTensor(), Normalize(0.5, 0.5)]),
        'test': Compose([ToTensor(), Normalize(0.5, 0.5)])
    },
    'byol': {
        'train': SimCLRTrainDataTransform(),
        'val': SimCLREvalDataTransform(),
        'test': SimCLREvalDataTransform()
    },
    'downstream_linear': {
        'train': SimCLREvalDataTransform(),
        'val': SimCLREvalDataTransform(),
        'test': SimCLREvalDataTransform()
    }
}


def transform_factory(model_name, mode):
    try:
        return transforms_dict.get(model_name).get(mode)
    except KeyError:
        raise NotImplementedError(f'{model_name} {mode} transform not implemented')

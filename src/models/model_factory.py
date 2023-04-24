from pl_bolts.models.self_supervised import SimCLR

from models.mae.model import MAE
from models.byol.model import BYOL


def model_factory(model_name, **model_kwargs):
    if model_name.lower() == 'simclr':
        return SimCLR(**model_kwargs)
    elif model_name.lower() == 'mae':
        return MAE
    elif model_name.lower() == 'byol':
        return BYOL(**model_kwargs)
    else:
        NotImplemented(f'Model {model_name} is not implemented!')


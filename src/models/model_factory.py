from pl_bolts.models.self_supervised import SimCLR, BYOL
from models.mae.model import MAE
from models.downstream.linear_eval import DownstreamClassifier

# from models.mae.model import MAE
# from models.byol.model import BYOL


def model_factory(model_name, **model_kwargs):
    if model_name.lower() == 'simclr':
        return SimCLR(**model_kwargs)
    elif model_name.lower() == 'mae':
        return MAE(**model_kwargs)
    elif model_name.lower() == 'byol':
        return BYOL(**model_kwargs)
    elif model_name.lower() == 'downstream_linear':
        return DownstreamClassifier(**model_kwargs)
    else:
        NotImplemented(f'Model {model_name} is not implemented!')


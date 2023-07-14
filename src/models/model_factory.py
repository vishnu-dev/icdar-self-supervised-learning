from pl_bolts.models.self_supervised import SimCLR
from src.models.byol.model import BYOL
from src.models.mae.model import MAE
from src.models.downstream.linear_eval import DownstreamClassifier


def model_factory(model_name, **model_kwargs):
    """Model factory for self-supervised models

    Args:
        model_name (str): Model name

    Returns:
        Any: Self-supervised model
    """
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


from pl_bolts.models.self_supervised import SimCLR
from src.models.mae.model import MAE
from src.models.byol.model import BYOL


def backbone_factory(model_name, checkpoint, **model_kwargs):
    """Backbone factory for self-supervised models

    Args:
        model_name (str): Backbone model name
        checkpoint (str): Checkpoint path

    Returns:
        Any: Backbone model with pre-trained weights
    """
    if model_name.lower() == 'simclr':
        model = SimCLR.load_from_checkpoint(checkpoint, **model_kwargs)
        return model.encoder
    elif model_name.lower() == 'mae':
        return MAE.load_from_checkpoint(checkpoint, **model_kwargs)
    elif model_name.lower() == 'byol':
        model = BYOL.load_from_checkpoint(checkpoint, **model_kwargs)
        return model.online_network
    else:
        NotImplemented(f'Model {model_name}:{checkpoint} checkpoint doest not exist!')


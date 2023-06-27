from pl_bolts.models.self_supervised import SimCLR, BYOL
from src.models.mae.model import MAE

# from models.mae.model import MAE
# from models.byol.model import BYOL


def backbone_factory(model_name, checkpoint, **model_kwargs):
    if model_name.lower() == 'simclr':
        model = SimCLR.load_from_checkpoint(checkpoint, **model_kwargs)
        return model.encoder
    elif model_name.lower() == 'mae':
        return MAE.load_from_checkpoint(checkpoint, **model_kwargs)
    elif model_name.lower() == 'byol':
        return BYOL.load_from_checkpoint(checkpoint, **model_kwargs)
    else:
        NotImplemented(f'Model {model_name}:{checkpoint} checkpoint doest not exist!')


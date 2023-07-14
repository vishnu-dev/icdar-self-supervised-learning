import pytorch_lightning as pl


def callback_factory(model_name):
    """Model callback factory

    Args:
        model_name (str): Model name

    Returns:
        List[Any]: List of callbacks
    """
    if model_name.lower() in ["simclr", "mae"]:
        return [pl.callbacks.ModelCheckpoint(mode="min", monitor="val_loss")]
    elif model_name.lower() == "byol":
        return [pl.callbacks.ModelCheckpoint(mode="min", monitor="val_loss")]
    elif model_name.lower() == "downstreamclassifier":
        return [pl.callbacks.ModelCheckpoint(mode="min", monitor="val_loss")]
    else:
        return None

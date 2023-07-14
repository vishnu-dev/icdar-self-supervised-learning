def collate_factory(model_name):
    """Custom collate function for each model.

    Args:
        model_name (str): Name of the model.

    Returns:
        Union[Callable, None]: Collate function for the model.
    """
    return None
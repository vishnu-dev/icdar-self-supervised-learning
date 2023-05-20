from lightly.data.multi_view_collate import MultiViewCollate


def collate_factory(model_name):
    if model_name.lower() == 'mae':
        return MultiViewCollate()
    else:
        return None
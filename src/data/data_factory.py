import os

from torch.utils.data import DataLoader, random_split
from data.dataset import ICDARDataset
from data.transforms import transform_factory
from lightly.data import LightlyDataset
from glob import glob


def data_factory(dataset_name, root_dir, label_filepath, transforms, mode, batch_size, collate_fn=None, num_cpus=None):

    if dataset_name.lower() == 'icdar':
        dataset = ICDARDataset(label_filepath, root_dir, transforms=transforms, convert_rgb=True)
    elif dataset_name.lower() == 'icdar_lightly':
        dataset = LightlyDataset(input_dir=root_dir, transform=transforms, filenames=glob(root_dir + '/*.tif'))
    else:
        raise NotImplementedError(f'Dataset {dataset_name} is not implemented')

    total_count = len(dataset)
    train_count = int(0.7 * total_count)
    val_count = int(0.1 * total_count)
    test_count = total_count - train_count - val_count

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        (train_count, val_count, test_count)
    )

    if mode in 'train':
        return {
            'train': DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
                pin_memory=True,
                persistent_workers=True,
                num_workers=num_cpus or os.cpu_count(),
                collate_fn=collate_fn if collate_fn else None
            ),
            'val': DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                pin_memory=True,
                persistent_workers=True,
                num_workers=num_cpus or os.cpu_count(),
                collate_fn=collate_fn if collate_fn else None
            )
        }
    elif mode == 'test':
        return {
            'test': DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                pin_memory=True,
                persistent_workers=True,
                num_workers=os.cpu_count(),
                collate_fn=collate_fn if collate_fn else None
            )
        }
    else:
        raise KeyError(f'Unknown mode: {mode}')

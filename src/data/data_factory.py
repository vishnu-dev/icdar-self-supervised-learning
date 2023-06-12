from torch.utils.data import DataLoader, random_split
from data.dataset import ICDARDataset
from lightly.data import LightlyDataset
from glob import glob
import os


def data_factory(
    dataset_name,
    root_dir,
    label_filepath,
    train_val_test_ratio,
    transforms,
    mode,
    batch_size,
    collate_fn=None,
    num_cpus=None,
    pin_memory=True
):
    
    print('Batch size: ', batch_size)

    if dataset_name.lower() == 'icdar':
        dataset = ICDARDataset(label_filepath, root_dir, transforms=transforms, convert_rgb=True)
    elif dataset_name.lower() == 'icdar_lightly':
        dataset = LightlyDataset(input_dir=root_dir, transform=transforms, filenames=glob(root_dir + '/*.tif'))
    else:
        raise NotImplementedError(f'Dataset {dataset_name} is not implemented')

    train_ratio, val_ratio, test_ratio = train_val_test_ratio

    total_count = len(dataset)
    train_count = int(train_ratio * total_count)
    val_count = int(val_ratio * total_count)
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
                pin_memory=pin_memory,
                num_workers=num_cpus or os.cpu_count(),
                prefetch_factor=2,
                collate_fn=collate_fn if collate_fn else None
            ),
            'val': DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                pin_memory=pin_memory,
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
                pin_memory=pin_memory,
                num_workers=os.cpu_count(),
                collate_fn=collate_fn if collate_fn else None
            )
        }
    else:
        raise KeyError(f'Unknown mode: {mode}')

from torch.utils.data import DataLoader, random_split
from src.data.dataset import ICDARDataset
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
    """Data loader factory based on dataset name.

    Args:
        dataset_name (str): Name of the dataset
        root_dir (str): Dataset root directory
        label_filepath (str): Label CSV filepath
        train_val_test_ratio (List[float]): List of ratios for train, val and test
        transforms (torchvision.transforms.Compose): Transforms to apply to the dataset
        mode (str): Execution mode (train, test)
        batch_size (int): Batch size
        collate_fn (Union[Callable, NoneType], optional): Collate function. Defaults to None.
        num_cpus (int, optional): Number of CPUs for data loading. Defaults to None.
        pin_memory (bool, optional): Whether to pin memory. Defaults to True.

    Raises:
        NotImplementedError: If dataset is not implemented
        KeyError: If defined mode is not implemented

    Returns:
        dict[str, torch.utils.data.DataLoader]: Dictionary of data loaders. Train and val for train mode, test for test mode
    """
    
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

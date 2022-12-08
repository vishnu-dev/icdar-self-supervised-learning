from dataset import load_icdar_task1_task3, ICDARDataset
import os

from image_utils import show_image_list


def test_run():
    proj_dir = 'shared/FAU Study/Project/PR Project'
    
    data_dir = os.path.join(
        os.path.expanduser('~'),
        proj_dir,
        'Data/ICDAR2017_CLaMM_task1_task3/'
    )
    
    label_file_path = os.path.join(
        data_dir,
        '@ICDAR2017_CLaMM_task1_task3.csv'
    )
    
    icdar_dataset = ICDARDataset(
        csv_filepath=label_file_path,
        root_dir=data_dir,
        transform=None
    )

    for i in range(len(icdar_dataset)):
        sample = icdar_dataset[i]
        show_image_list([sample.get('image')])

    # load_icdar_task1_task3(data_dir, label_file_path, 10, show_samples_num=5)


if __name__ == '__main__':
    test_run()
    

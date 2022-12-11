from augment import PositivePairTransform
from dataset import ICDARDataset
import os
import torchvision.transforms as T
from image_utils import show_image_list
from tqdm import tqdm


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

    num_images_show = 10
    
    icdar_dataset = ICDARDataset(
        csv_filepath=label_file_path,
        root_dir=data_dir,
        transform=PositivePairTransform()
    )

    images = []
    labels = []
    
    for i in tqdm(range(num_images_show)):
        sample = icdar_dataset[i]
        std_img = T.RandomResizedCrop((512, 512))(sample['image'])
        images.append(std_img[0, :].numpy())
        labels.append(sample['positive_pair'])

    show_image_list(images, grid=False, num_cols=3, list_titles=labels)


if __name__ == '__main__':
    test_run()
    

import torch.nn
import torch.nn.functional as F


class ContrastiveLoss(torch.nn.Module):
    
    def __init__(self, batch_size, temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature
    
    @staticmethod
    def similarity(left, right):
        # Batch wise similarity
        representations = torch.cat([left, right], dim=0)
        return F.cosine_similarity(
            representations.unsqueeze(1),
            representations.unsqueeze(0),
            dim=2
        )
    
    def forward(self, left, right):
        batch_size = left.shape[0]
        
        # L2 norm
        left_norm = F.normalize(left, p=2, dim=1)
        right_norm = F.normalize(right, p=2, dim=1)

        # Cosine similarity
        similarity_matrix = self.similarity(left_norm, right_norm)

        # Get all positives
        positives_lower_triangle = torch.diag(similarity_matrix, batch_size)
        positives_upper_triangle = torch.diag(similarity_matrix, -batch_size)
        positives = torch.cat([
            positives_lower_triangle, positives_upper_triangle
        ], dim=0)

        # Mask for positives and negatives
        mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=torch.bool)).float()

        # Apply the vanilla contrastive loss formula
        numerator = torch.exp(positives / self.temperature)
        denominator = mask.to(similarity_matrix) * torch.exp(similarity_matrix / self.temperature)
        losses = -torch.log(numerator / torch.sum(denominator, dim=1))
        
        # Return average loss (2 * batch_size because of positive-negative pairs)
        avg_loss = torch.sum(losses) / (2 * self.batch_size)
        return avg_loss


def test():
    
    import os
    from src.dataset import ICDARDataset
    from src.augment import PositivePairTransform
    import torchvision.transforms as T
    
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
    
    num_images_show = 2
    
    icdar_dataset = ICDARDataset(
        csv_filepath=label_file_path,
        root_dir=data_dir
    )
    transforms = PositivePairTransform()
    contrastive_loss = ContrastiveLoss(1)
    
    batch_images_left, batch_images_right = [], []
    
    for i in range(num_images_show):
        sample = icdar_dataset[i]
        std_img = T.RandomResizedCrop((512, 512))(sample['image'])
        batch_images_left.append(transforms.left(std_img))
        batch_images_right.append(transforms.right(std_img))

    batch_left = torch.cat(batch_images_left, dim=0)
    batch_right = torch.cat(batch_images_right, dim=0)
    
    print(contrastive_loss(batch_left, batch_right))


if __name__ == '__main__':
    test()

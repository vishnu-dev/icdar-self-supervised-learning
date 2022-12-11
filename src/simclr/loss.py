import torch.nn
import torch.nn.functional as F


class ContrastiveLoss(torch.nn.Module):
    
    def __init__(self, batch_size, temperature):
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

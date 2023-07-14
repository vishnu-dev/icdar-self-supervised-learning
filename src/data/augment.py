import torchvision.transforms as T
import torch
from kornia.morphology import erosion, dilation


class GaussianNoise(object):
    
    def __init__(self, mean=0., std=1.):
        """Gaussian noise transform.

        Args:
            mean (float, optional): Mean of the distribution for noise. Defaults to 0.0.
            std (_type_, optional): Standard deviation of the distribution for noise. Defaults to 1.0.
        """
        self.std = std
        self.mean = mean
    
    def __call__(self, tensor):
        out = tensor + torch.randn(tensor.size()) * self.std + self.mean
        return out
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class Erosion(object):
    
    def __init__(self, kernel_size=3):
        """Erosion transform.
        
        Args:
            kernel_size (int, optional): Kernel size for erosion. Defaults to 3.
        """
        self.kernel_size = kernel_size
    
    def __call__(self, tensor):
        erosion_kernel = torch.ones(self.kernel_size, self.kernel_size)
        batch_tensor = tensor[None, :, :, :]
        batch_tensor = erosion(batch_tensor, erosion_kernel)
        return batch_tensor[0, :, :, :]
    

class Dilation(object):
    
    def __init__(self, kernel_size=3):
        """Dilation transform.

        Args:
            kernel_size (int, optional): Kernel size for dilation. Defaults to 3.
        """
        self.kernel_size = kernel_size
    
    def __call__(self, tensor):
        erosion_kernel = torch.ones(self.kernel_size, self.kernel_size)
        structuring_kernel = torch.zeros(self.kernel_size, self.kernel_size)
        batch_tensor = tensor[None, :, :, :]
        batch_tensor = dilation(batch_tensor, erosion_kernel, structuring_element=structuring_kernel)
        return batch_tensor[0, :, :, :]


class PairTransform:
    
    def __init__(self, transforms=None, online_transforms=None):
        """Pair transform for contrastive models.
        
        Args:
            transforms (torchvision.transforms, optional): Transforms for the pair. Defaults to None.
            online_transforms (torchvision.transforms, optional): Transforms for the online image. Defaults to None.
        """
        self.transforms = transforms
        self.online_transform = online_transforms
    
    def __call__(self, tensor):
        left = self.transforms(tensor)
        right = self.transforms(tensor)
        online = self.online_transform(tensor)
        return left, right, online

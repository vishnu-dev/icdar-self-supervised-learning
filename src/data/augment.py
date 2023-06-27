import torchvision.transforms as T
import torch
from kornia.morphology import erosion, dilation


class GaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
    
    def __call__(self, tensor):
        out = tensor + torch.randn(tensor.size()) * self.std + self.mean
        return out
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class Erosion(object):
    
    def __init__(self, kernel_size=3):
        self.kernel_size = kernel_size
    
    def __call__(self, tensor):
        erosion_kernel = torch.ones(self.kernel_size, self.kernel_size)
        batch_tensor = tensor[None, :, :, :]
        batch_tensor = erosion(batch_tensor, erosion_kernel)
        return batch_tensor[0, :, :, :]
    

class Dilation(object):
    
    def __init__(self, kernel_size=3):
        self.kernel_size = kernel_size
    
    def __call__(self, tensor):
        erosion_kernel = torch.ones(self.kernel_size, self.kernel_size)
        structuring_kernel = torch.zeros(self.kernel_size, self.kernel_size)
        batch_tensor = tensor[None, :, :, :]
        batch_tensor = dilation(batch_tensor, erosion_kernel, structuring_element=structuring_kernel)
        return batch_tensor[0, :, :, :]


class NegativePairTransform:
    
    def __init__(self, transforms=None, online_transforms=None):
        self.transforms = transforms
        self.online_transform = online_transforms
    
    def __call__(self, tensor):
        left = self.transforms(tensor)
        right = self.transforms(tensor)
        online = self.online_transform(tensor)
        return left, right, online

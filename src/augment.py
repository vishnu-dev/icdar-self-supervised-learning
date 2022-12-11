import random
import numpy as np
import torchvision.transforms as T
import torch
import cv2 as cv
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


class PositivePairTransform:
    
    def __init__(self, transforms=None):
        self.transforms = (
            transforms if transforms is not None else
            [
                # T.RandomCrop(254),
                GaussianNoise(0, random.uniform(0.1, 0.3)),
                T.GaussianBlur(kernel_size=(5, 5)),
                T.RandomRotation(degrees=np.random.randint(0, 45)),
                T.RandomErasing(p=1),
                Dilation(5),
            ]
        )
    
    def __call__(self, tensor):
        self.left, self.right = random.sample(self.transforms, 2)
        tensor = self.left(tensor)
        tensor = self.right(tensor)
        return tensor
    
    def __repr__(self):
        return f"{type(self.left).__name__} + {type(self.right).__name__}"
    
    def __str__(self):
        return f"{type(self.left).__name__} + {type(self.right).__name__}"


if __name__ == '__main__':
    from PIL import Image
    import matplotlib.pyplot as plt
    
    image_path = "/Users/vishnudev/shared/FAU Study/Project/PR " \
                 "Project/Data/ICDAR2017_CLaMM_task1_task3/315556101_MS0364_0077.tif"
    im = Image.open(image_path)
    image_tensor = T.ToTensor()(im)
    pp = PositivePairTransform()
    im = pp(image_tensor)
    im = T.RandomResizedCrop((256, 256))(im)
    plt.imshow(im[0, :].numpy(), cmap='gray')
    plt.show()

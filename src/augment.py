import numpy as np
import torchvision.transforms as T


class Augment:
    
    def __init__(self, image):
        self.im = image
    
    def apply(self, num_comp=1):
        
        augmentations = [
            self.crop, self.rotate, self.cut_out, self.color, self.sobel, self.noise, self.blur
        ]
        applied = []
        aug_im = None
        
        for comp in range(num_comp):
            rand_fun = np.random.choice(augmentations)
            aug_im = rand_fun()
            applied.append(rand_fun)

        return aug_im
    
    def crop(self):
        # TODO: Random crop based on image size
        crop_image = T.RandomCrop(size=np.random.choice([100, 200, 300]))(self.im)
        return crop_image

    def rotate(self):
        return T.RandomRotation(degrees=np.random.choice(range(360)))(self.im)
    
    def cut_out(self):
        return T.RandomErasing()(self.im)
    
    def color(self):
        pass
    
    def sobel(self):
        pass
    
    def noise(self):
        pass
    
    def blur(self):
        sigma = np.random.choice([3, 5, 7])
        return T.GaussianBlur(kernel_size=(11, 11), sigma=sigma)(self.im)


if __name__ == '__main__':
    from PIL import Image
    import matplotlib.pyplot as plt
    
    image_path = "/Users/vishnudev/shared/FAU Study/Project/PR " \
                 "Project/Data/ICDAR2017_CLaMM_task1_task3/315556101_MS0364_0077.tif"
    im = Image.open(image_path)
    image_tensor = T.ToTensor()(im)
    augment = Augment(image_tensor)
    im = augment.cut_out()

    plt.imshow(np.squeeze(im.permute(1, 2, 0), axis=2), cmap='gray')
    plt.show()
    
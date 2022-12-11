from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as T


class ResNet50Encoder(object):
    
    def __init__(self):
        self.weights = ResNet50_Weights.DEFAULT
        self.model = resnet50(weights=self.weights)
        self.model.eval()

    def __call__(self, tensor):
        tensor = tensor.unsqueeze(0)
        tensor = tensor.repeat(1, 3, 1, 1)
    
        preprocess = self.weights.transforms()
        tensor = preprocess(tensor)
        prediction = self.model(tensor)
    
        return prediction
    

def test():
    from PIL import Image
    import matplotlib.pyplot as plt
    
    image_path = "/Users/vishnudev/shared/FAU Study/Project/PR " \
                 "Project/Data/ICDAR2017_CLaMM_task1_task3/315556101_MS0364_0077.tif"

    im = Image.open(image_path)
    im = T.ToTensor()(im)
    im = T.RandomResizedCrop((256, 256))(im)
    vector = ResNet50Encoder()(im)
    print(vector.shape)
    # plt.imshow(im[0, :].numpy(), cmap='gray')
    # plt.show()


if __name__ == '__main__':
    test()

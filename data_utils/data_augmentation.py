
import numpy as np
from scipy import ndimage




class RandomFlip2D(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        mm = 1 if len(image.shape) > 2 else 0

        k = np.random.randint(0, 4)
        axis = np.random.randint(0, 2)
        if mm:
            for i in range(image.shape[0]):
                image[i] = np.flip(np.rot90(image[i], k), axis=axis).copy()
        else:
            image = np.flip(np.rot90(image, k), axis=axis).copy()
        label = np.flip(np.rot90(label, k), axis=axis).copy()
        return {'image': image, 'label': label}


class RandomRotate2D(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """
    def __init__(self, degree=(-20, 20)):
        assert len(degree) == 2
        self.degree = degree

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        mm = 1 if len(image.shape) > 2 else 0
        angle = np.random.randint(self.degree[0], self.degree[1])
        if mm:
            for i in range(image.shape[0]):
                image[i] = ndimage.rotate(image[i], angle, order=0, reshape=False)
        else:
            image = ndimage.rotate(image, angle, order=0, reshape=False)
        label = ndimage.rotate(label, angle, order=0, reshape=False)

        return {'image': image, 'label': label}








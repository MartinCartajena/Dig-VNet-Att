import numpy as np
from skimage.transform import rescale, rotate
from torchvision.transforms import Compose


def transforms(angle=None, horizontal_flip_prob=None, vertical_flip_prob=None, salt_pepper_prob=None):
    transform_list = []

    if angle is not None:
        transform_list.append(Rotate(angle))
    if horizontal_flip_prob is not None:
        transform_list.append(HorizontalFlip(horizontal_flip_prob))
    if vertical_flip_prob is not None:
        transform_list.append(VerticalFlip(vertical_flip_prob))
    if salt_pepper_prob is not None:
        transform_list.append(SaltAndPepper(prob=salt_pepper_prob['prob'], amount=salt_pepper_prob['amount'], salt_ratio=salt_pepper_prob['salt_ratio']))

    return Compose(transform_list)


class Rotate(object):

    def __init__(self, angle):
        self.angle = angle

    def __call__(self, sample):
        image, mask = sample

        angle = np.random.uniform(low=-self.angle, high=self.angle)
        image = rotate(image, angle, resize=False, preserve_range=True, mode="constant")
        mask = rotate(
            mask, angle, resize=False, order=0, preserve_range=True, mode="constant"
        )
        return image, mask


class HorizontalFlip(object):

    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, sample):
        image, mask = sample

        if np.random.rand() > self.flip_prob:
            return image, mask

        image = np.fliplr(image).copy()
        mask = np.fliplr(mask).copy()

        return image, mask
    
    
class VerticalFlip(object):
    
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, sample):
        image, mask = sample

        if np.random.rand() > self.flip_prob:
            return image, mask

        image = np.flipud(image).copy()
        mask = np.flipud(mask).copy()

        return image, mask


class SaltAndPepper(object):
    
    def __init__(self, prob, amount, salt_ratio):
        self.prob = prob
        self.amount = amount
        self.salt_ratio = salt_ratio

    def __call__(self, sample):
        image, mask = sample
        
        if np.random.rand() > self.prob:
            return image, mask
        
        num_pixels = int(self.amount * image.size)

        # Salt noise (white pixels)
        num_salt = int(self.salt_ratio * num_pixels)
        coords_salt = [np.random.randint(0, i, num_salt) for i in image.shape]
        image[coords_salt] = 1

        # Pepper noise (black pixels)
        num_pepper = num_pixels - num_salt
        coords_pepper = [np.random.randint(0, i, num_pepper) for i in image.shape]
        image[coords_pepper] = 0

        return image, mask


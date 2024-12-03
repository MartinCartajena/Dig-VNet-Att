import numpy as np
from skimage.transform import rescale, rotate
from torchvision.transforms import Compose


def transforms(rot_prob=None, horizontal_flip_prob=None, vertical_flip_prob=None, salt_pepper_prob=None):
    transform_list = []

    if rot_prob is not None:
        transform_list.append(Rotate(rot_prob))
    if horizontal_flip_prob is not None:
        transform_list.append(HorizontalFlip(horizontal_flip_prob))
    if vertical_flip_prob is not None:
        transform_list.append(VerticalFlip(vertical_flip_prob))
    if salt_pepper_prob is not None:
        transform_list.append(SaltAndPepper(prob=salt_pepper_prob['prob'], amount=salt_pepper_prob['amount'], salt_ratio=salt_pepper_prob['salt_ratio']))

    return Compose(transform_list)


class Rotate(object):

    def __init__(self, rot_prob):
        self.rot_prob = rot_prob
        self.angle = 90 

    def __call__(self, sample):
        image, mask = sample
        
        if np.random.rand() > self.rot_prob:
            return image, mask
        
        angle = (int(np.random.rand()) + 1) * self.angle
        
         # Rotamos sobre los ejes y, z (dimensiones 1 y 2) sin afectar la dimensión 0
        image_rotated = np.array([
            rotate(slice_2d, angle, resize=False, preserve_range=True, mode="constant")
            for slice_2d in image
        ])
        mask_rotated = np.array([
            rotate(slice_2d, angle, resize=False, order=0, preserve_range=True, mode="constant")
            for slice_2d in mask
        ])

        # Garantizamos que la forma de las imágenes no cambia recortándolas al tamaño original
        image_cropped = image_rotated[:, :image.shape[1], :image.shape[2]]
        mask_cropped = mask_rotated[:, :mask.shape[1], :mask.shape[2]]

        return image_cropped, mask_cropped


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
        image[tuple(coords_salt)] = 1

        # Pepper noise (black pixels)
        num_pepper = num_pixels - num_salt
        coords_pepper = [np.random.randint(0, i, num_pepper) for i in image.shape]
        image[tuple(coords_pepper)] = 0

        return image, mask


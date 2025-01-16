import os
import numpy as np
import torch
from torchvision.transforms import Compose, Normalize, ToTensor, Resize
from scipy.ndimage import zoom
from utils.datasets.noduleVoxels.guardado import prepro

from utils.prepare.data_augmentation import DataAugmentation

# width, height = 96, 96

def att_compare(a, b=3):
    if a > b:
        return np.array([1])
    else:
        return np.array([0])


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def normalazation(image_array):
    max = image_array.max()
    min = image_array.min()
    image_array = (image_array-min)/(max-min)  # float cannot apply the compute,or array error will occur
    avg = image_array.mean()
    image_array = image_array-avg
    return image_array   # a bug here, a array must be returned,directly appling function did't work
    

class Dataset(torch.utils.data.Dataset):
    def __init__(self, datas, noduleid, data_aug =False, transform=None, label=None, width=64, height=64, depth = 16, use_cache = False):
        self.size = (depth, width, height)
        self.datas = datas
        self.id = noduleid
        self.cached_data = []
        self.cached_inters = []
        self.cached_unions = []
        self.cached_masks = []
        self.cached_inputs = []
        self.use_cache = use_cache
        self.data_aug = data_aug
        self.transform = transform

        if data_aug:
            self.data_aug = DataAugmentation(noise_amount=0.03, salt_ratio=0.5, orientation="LPS")

        self.img_transform_gray = Compose([
            Normalize(mean=[0.448749], std=[0.399953])  # 这个归一化方式可以提升近1个点的dice
        ])

        self.img_transform_rgb = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.input_paths = self.datas

    def __getitem__(self, index):
        if not self.use_cache:
            input_path = self.input_paths[index]
            noduleid = self.id[index]

            im, seg = prepro(input_path, noduleid)

            scaling_factors = (16/im.shape[0], 64/im.shape[1], 64/im.shape[2])

            input = zoom(im, scaling_factors, order = 1)
            mask = zoom(seg, scaling_factors, order = 1)

            #Meter canales
            # input = input.unsqueeze(0)
            # mask = mask.unsqueeze(0)
            mask[mask > 0.5] = 1
            mask[mask < 0.5] = 0
                            
            input_original = torch.from_numpy(input).float()
            mask_original = torch.from_numpy(mask).float()
            
            input_original = self.img_transform_gray(input_original)
            mask_original = self.img_transform_gray(mask_original)
            
            self.cached_data.append(input_original)
            self.cached_masks.append(mask_original) 
            
            if self.data_aug:
                input_aug, label_aug = self.data_aug(input, mask)
                
                if len(input_aug) > 0:
                    for i in input_aug: 
                                
                        input = torch.from_numpy(input_aug[i].copy()).float()
                        mask = torch.from_numpy(label_aug[i.replace("_image","_mask")].copy()).float()
                        
                        input = self.img_transform_gray(input)
                        mask = self.img_transform_gray(mask)
                        
                        mask[mask > 0.5] = 1
                        mask[mask < 0.5] = 0
                        
                        self.cached_data.append(input)
                        self.cached_masks.append(mask)  
                                       
        else:
            input = self.cached_data[index]
            mask = self.cached_masks[index]
            
            if self.transform:
                input = input.cpu().numpy()
                mask = mask.cpu().numpy()
                input, mask = self.transform((input, mask))
                
                input = torch.from_numpy(input).float()
                mask = torch.from_numpy(mask).long()

        mask[mask > 0.5] = 1
        mask[mask < 0.5] = 0
        
        file_name = input_path.split("/")[len(input_path.split("/"))-1]
        
        return input, file_name, mask

    def __len__(self):
        if self.use_cache and self.data_aug:
            return len(self.input_paths) * 11
        else:
            return len(self.input_paths)
        
    def set_use_cache(self, use_cache):
        # if use_cache:
        #     x_img = tuple(self.cached_data)
        #     self.cached_data = torch.stack(x_img)
        #     x_seg = tuple(self.cached_masks)
        #     self.cached_masks = torch.stack(x_seg)
        #     x_int = tuple(self.cached_inters)
        #     self.cached_inters = torch.stack(x_int)
        #     x_uni = tuple(self.cached_unions)
        #     self.cached_unions = torch.stack(x_uni)
        # else: 
        #     self.cached_data = []
        #     self.cached_inters = []
        #     self.cached_masks = []
        #     self.cached_unions = []

        self.use_cache = use_cache

    def getCache(self):
        return self.cached_data



def loader(dataset, batch_size, num_workers=4, shuffle=False):
    input_images = dataset

    input_loader = torch.utils.data.DataLoader(dataset=input_images,
                                               batch_size=batch_size,
                                               shuffle=shuffle,
                                               num_workers=num_workers)

    return input_loader

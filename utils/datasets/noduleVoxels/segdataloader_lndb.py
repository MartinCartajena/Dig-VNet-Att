import json
import os
import numpy as np
import torch
from torchvision.transforms import Compose, Normalize, ToTensor, Resize
from scipy.ndimage import zoom
# from needed.dataprocess.guardado import prepro
import pydicom

# width, height = 96, 96


def att_compare(a, b=3):
    if a > b:
        return np.array([1])
    else:
        return np.array([0])


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def normalazation(image_array):
    max = image_array.max()
    min = image_array.min()
    image_array = (image_array-min)/(max-min)  # float cannot apply the compute,or array error will occur
    avg = image_array.mean()
    image_array = image_array-avg
    return image_array   # a bug here, a array must be returned,directly appling function did't work

class Dataset(torch.utils.data.Dataset):
    def __init__(self, datas, noduleid, label=None, width=64, height=64, depth = 16, use_cache = False):
        self.size = (depth, width, height)
        self.datas = datas
        self.id = noduleid
        self.cached_data = []
        self.cached_inters = []
        self.cached_unions = []
        self.cached_masks = []
        self.cached_inputs = []
        self.use_cache = use_cache


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

            im, seg, red, blu = prepro(input_path, noduleid)

            scaling_factors = (16/im.shape[0], 64/im.shape[1], 64/im.shape[2])

            input = zoom(im, scaling_factors, order = 1)
            inter = zoom(blu, scaling_factors, order = 1) 
            union = zoom(red, scaling_factors, order = 1)
            mask = zoom(seg, scaling_factors, order = 1)

            nombre = input_path.split('/')[3] + '_' + str(noduleid[3]) + '_' + str(noduleid[2])

        

            input = torch.from_numpy(input).float()
            inter = torch.from_numpy(inter).float()
            union = torch.from_numpy(union).float()
            mask = torch.from_numpy(mask).float()

            input = self.img_transform_gray(input)
            inter = self.img_transform_gray(inter)
            union = self.img_transform_gray(union)
            mask = self.img_transform_gray(mask)

            #Meter canales
            input = input.unsqueeze(0)
            inter = inter.unsqueeze(0)
            union = union.unsqueeze(0)
            mask = mask.unsqueeze(0)

        

            # 二值化
            inter[inter > 0.5] = 1
            inter[inter < 0.5] = 0
            union[union > 0.5] = 1
            union[union < 0.5] = 0
            mask[mask > 0.5] = 1
            mask[mask < 0.5] = 0

            self.cached_data.append(input)
            self.cached_inters.append(inter)
            self.cached_unions.append(union)
            self.cached_masks.append(mask)
            self.cached_inputs.append(input_path)

        else:
            input = self.cached_data[index]
            union = self.cached_unions[index]
            inter = self.cached_inters[index]
            mask = self.cached_masks[index]


        return input, union, inter, mask, nombre, im.shape

    def __len__(self):
        return len(self.input_paths)
    
    def set_use_cache(self, use_cache):
        if use_cache:
            x_img = tuple(self.cached_data)
            self.cached_data = torch.stack(x_img)
            x_seg = tuple(self.cached_masks)
            self.cached_masks = torch.stack(x_seg)
            x_int = tuple(self.cached_inters)
            self.cached_inters = torch.stack(x_int)
            x_uni = tuple(self.cached_unions)
            self.cached_unions = torch.stack(x_uni)
        else: 
            self.cached_data = []
            self.cached_inters = []
            self.cached_masks = []
            self.cached_unions = []

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

class DatasetLNDb(torch.utils.data.Dataset):
    def __init__(self, datas, label, inter, union, width=64, height=64, depth = 16):
        self.size = (depth, width, height)
        self.datas = datas
        self.inters = inter
        self.unions = union
        self.masks = label


        self.img_transform_gray = Compose([
            Normalize(mean=[0.448749], std=[0.399953])  # 这个归一化方式可以提升近1个点的dice
        ])

        self.img_transform_rgb = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.input_paths = self.datas

    def __getitem__(self, index):
            input_path = self.input_paths[index]
            im = np.load(input_path)
            blu = np.load(self.inters[index])
            red = np.load(self.unions[index])
            seg = np.load(self.masks[index])
            scaling_factors = (16/im.shape[0], 64/im.shape[1], 64/im.shape[2])

            im = ((im - np.min(im))) / (np.max(im) - np.min(im))
            im = (im * 255).astype(np.uint8)

            input = zoom(im, scaling_factors, order = 1)
            inter = zoom(blu, scaling_factors, order = 1) 
            union = zoom(red, scaling_factors, order = 1)
            mask = zoom(seg, scaling_factors, order = 1)

            input = torch.from_numpy(input).float()
            inter = torch.from_numpy(inter).float()
            union = torch.from_numpy(union).float()
            mask = torch.from_numpy(mask).float()

            # input = self.img_transform_gray(input)
            # inter = self.img_transform_gray(inter)
            # union = self.img_transform_gray(union)
            # mask = self.img_transform_gray(mask)

            #Meter canales
            input = input.unsqueeze(0)
            inter = inter.unsqueeze(0)
            union = union.unsqueeze(0)
            mask = mask.unsqueeze(0)

            nombre = input_path.split('/')[-1]

        

            # 二值化
            inter[inter > 0.5] = 1
            inter[inter < 0.5] = 0
            union[union > 0.5] = 1
            union[union < 0.5] = 0
            mask[mask > 0.5] = 1
            mask[mask < 0.5] = 0


            return input, nombre, mask

    def __len__(self):
        return len(self.input_paths)

class DatasetNLST(torch.utils.data.Dataset):
    def __init__(self, datas, bbs, numero, width=64, height=64, depth = 16):
        self.size = (depth, width, height)
        self.datas = datas
        self.bbs = bbs
        self.numero = numero



        self.img_transform_gray = Compose([
            Normalize(mean=[0.448749], std=[0.399953])  # 这个归一化方式可以提升近1个点的dice
        ])

        self.img_transform_rgb = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.input_paths = self.datas

    def __getitem__(self, index):
            input_path = self.input_paths[index]
            bb = self.bbs[index]
            I = input_path.split('/')
            path = '/app/imagenes/' + '/'.join(I[8:])
            dcms = os.listdir(path)
            length = bb['instance_num'][-1] - bb['instance_num'][0] + 1
            im = []
            for i in range(length):
                if i == 0:
                    ds = pydicom.dcmread(path + bb['ini_slice'] + '.dcm')
                    im.append(ds.pixel_array[int(bb['ymin']):int(bb['ymax'])+1, int(bb['xmin']):int(bb['xmax'])+1])
                    inst = bb['instance_num'][0]
                else:
                    for j in dcms:
                        ds = pydicom.dcmread(path + j)
                        if ds.InstanceNumber == inst+1:
                            im.append(ds.pixel_array[int(bb['ymin']):int(bb['ymax'])+1, int(bb['xmin']):int(bb['xmax'])+1])
                            inst = inst+1
                            break
            im = np.array(im)

            scaling_factors = (16/im.shape[0], 64/im.shape[1], 64/im.shape[2])

            im = ((im - np.min(im))) / (np.max(im) - np.min(im))
            im = (im * 255).astype(np.uint8)

            input = zoom(im, scaling_factors, order = 1)

            input = torch.from_numpy(input).float()

            input = self.img_transform_gray(input)

            #Meter canales
            input = input.unsqueeze(0)

            nombre = input_path.split('/')[-2] + '_' + str(self.numero[index])
            center = [(bb['xmax'] - bb['xmin'])/2, (bb['ymax'] - bb['ymin'])/2]


            return input, im.shape, nombre, center

    def __len__(self):
        return len(self.input_paths)


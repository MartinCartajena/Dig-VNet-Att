'''
Created by Wang Qiu Li
7/3/2018

get dicom info according to malignancy.csv and ld_scan.txt
'''

import csv
import os
import pydicom
import imageio
import numpy as np
import tqdm
import pandas as pd
from matplotlib.pyplot import imread
import random

import utils.datasets.noduleVoxels.xmlopt as xmlopt


def readCSV(filename):
    lines = []
    with open(filename, "r") as f:
        csvreader = csv.reader(f)
        for line in csvreader:
            lines.append(line)
    return lines

def get_pixels_hu(ds):
    image = ds.pixel_array
    image = np.array(image , dtype = np.float32)
    intercept = ds.RescaleIntercept
    slope = ds.RescaleSlope
    image = image * slope
    image += intercept
    return image

def getThreeChannel(pixhu):
    lungwindow = truncate_hu(pixhu, 800, -1000)
    highattenuation = truncate_hu(pixhu, 240, -160)
    lowattenuation = truncate_hu(pixhu, -950, -1400)
    pngfile = [lungwindow, highattenuation, lowattenuation]
    pngfile = np.array(pngfile).transpose(1,2,0)
    return  pngfile  

def truncate_hu(image_array, max, min):
    image = image_array.copy()
    image[image > max] = max
    image[image < min] = min
    image = normalazation(image)
    return image
    
# LUNA2016 data prepare ,second step: normalzation the HU
def normalazation(image_array):
    max = image_array.max()
    min = image_array.min()
    image_array = (image_array-min)/(max-min)  # float cannot apply the compute,or array error will occur
    avg = image_array.mean()
    image_array = image_array-avg
    return image_array   # a bug here, a array must be returned,directly appling function did't work


def cutTheImage(xmin, xmax, ymin, ymax, pix):
    img_cut = pix[ymin:ymax,xmin:xmax]
    return img_cut

def caseid_to_scanid(caseid):
    returnstr = ''
    if caseid < 10:
        returnstr = '000' + str(caseid)
    elif caseid < 100:
        returnstr = '00' + str(caseid)
    elif caseid < 1000:
        returnstr = '0' + str(caseid)
    else:
        returnstr = str(caseid)
    return 'LIDC-IDRI-' + returnstr

def reverse(inputarray):
    shape = inputarray.shape
    nparray = np.ones(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            if inputarray[i][j] == 0:
                nparray[i][j] = 1
            else:
                nparray[i][j] = 0
    return nparray

def red_mask(image_list):
    if len(image_list) == 1:
        return image_list[0]
    else:
        temp = np.zeros((512,512))
        for i in range(len(image_list)):
            temp += image_list[i]
        temp[temp > 0.5] = 1.0
        return temp

def blue_mask(image_list):
    if len(image_list) == 1:
        return image_list[0]
    else:
        temp = np.zeros((512,512))
        for i in range(len(image_list)):
            temp += image_list[i]
        temp[temp < len(image_list)] = 0.0
        return temp

def dif_mask(image_list):
    if len(image_list) == 1:
        return np.zeros((512,512))
    else:
        temp = np.zeros((512,512))
        for i in range(len(image_list)):
            temp += image_list[i]
        temp[temp == len(image_list)] = 0.0
        temp[temp > 0] = 1.0

        return temp

def prepro(scanpaths, noduleinfo):
    onenodule = noduleinfo
    xml = ''
    scanid = onenodule[1]
    scanid = caseid_to_scanid(int(scanid))
    noduleid = onenodule[3]

    noduleld_list = []
    for i in range(10, 14):
        if str(onenodule[i]).strip() != '':
            noduleld_list.append(onenodule[i])

    filelist1 = os.listdir(scanpaths)
    filelist2 = []

    xmlfiles = []
    for onefile in filelist1:
        if '.dcm' in onefile:
            filelist2.append(onefile)
        elif '.xml' in onefile:
            xmlfiles= onefile

    xmlfile = scanpaths + '/' + xmlfiles
    
    slices = []
    j = 0
  

    slices = [pydicom.dcmread(scanpaths + '/' + s) for s in filelist2]
    slices.sort(key = lambda x : float(x.ImagePositionPatient[2]),reverse=True)
    z_loc = int(onenodule[8])
    ds = slices[z_loc - 1]

    masks = []

    if (str(ds.SeriesNumber) == str(onenodule[2])) or (str(onenodule[2]) == str(0)):

        slice_location = ds.ImagePositionPatient[2]
        space = float(ds.SliceThickness)
        red = []
        blue = []
        dif = []
        pos_acc = []
        for one_nodule in noduleld_list:
            mask_image, signtempm, pos = xmlopt.getEdgeMap(xmlfile, slice_location, [one_nodule]) #saca mascaras de las imagenes
            masks.append(mask_image)
            pos_acc.append(pos)
        positions = []
        count = []

        for i in range(len(pos_acc)):
            if i == 0:
                for slice in pos_acc[i]:
                    positions.append(float(slice))
                    count.append(1)
            for A in pos_acc[i]:
                if float(A) not in positions and i!=len(pos_acc)-1:
                    positions.append(float(A))
                    count.append(1)
                elif float(A) in positions and i !=0:
                    count[positions.index(float(A))] +=1
        sub1 = []
        for i in range(len(positions)):
            if count[i]==1:
                sub1.append(i)
        for i in sorted(sub1, reverse=True):
            del positions[i]
            del count[i]
        
        


        index = []
        for i in positions:
            ind=[]
            for A in pos_acc:
                if i in A:
                    ind.append(A.index(i))
                else:
                    ind.append(-5)
            index.append(ind)

        masks_sorted = []
        for i in range(len(positions)):
            mask_pos = []
            for j in range(len(masks)):
                if index[i][j]>-1:
                    A = masks[j][index[i][j]]
                    mask_pos.append(A)
            masks_sorted.append(mask_pos)

        addz1 = random.randint(0,5)
        addz2 = random.randint(0,5)
        im = []
        for i in range(len(masks_sorted[0])):
            im.append(np.zeros((512,512)))

        for i in range(addz1):
            valor = positions[0]+space
            positions.insert(0,valor)
            masks_sorted.insert(0,im)
        for i in range(addz2):
            valor = positions[-1]-space
            positions.append(valor)
            masks_sorted.append(im)

        for i in range(len(masks_sorted)):
            red.append(red_mask(masks_sorted[i]))
            blue.append(blue_mask(masks_sorted[i]))
            dif.append(dif_mask(masks_sorted[i]))

        index_zloc1 = positions.index(slice_location)
        lenpos = len(positions)
        images = [np.zeros((512, 512)) for _ in range(lenpos)]   
        for i in range(lenpos+40):
            if z_loc - int(lenpos/2) + i - 20<0 or z_loc - int(lenpos/2) + i - 20>(len(slices)-1):
                continue
            ds1 = slices[z_loc - int(lenpos/2) + i - 20]
            location = float(ds1.ImagePositionPatient[2])
            if location in positions:
                ind = positions.index(location)
                ori_hu=get_pixels_hu(ds1)
                images[ind] = ori_hu
        # imageio.imsave(imagedir + str(scanid) + '_' + str(noduleid) + '_' + str(scan_list_id) + '_' + str(positions[i]) +'_original.png', images[0])
        xmins = []
        xmaxs = []
        ymins = []
        ymaxs = []
        for i in range(lenpos):
            mask_image = masks_sorted[i][0]
            arrays = np.where(mask_image==1)
            if len(arrays[0])==0:
                continue
            xmins.append(np.min(arrays[1]))
            ymins.append(np.min(arrays[0]))
            xmaxs.append(np.max(arrays[1]))
            ymaxs.append(np.max(arrays[0]))
        
        xmin = np.min(xmins)
        xmax = np.max(xmaxs)
        ymin = np.min(ymins)
        ymax = np.max(ymaxs)

        img = []
        seg = []
        blues = []
        reds = []
        tempx1 = random.randint(5,15)
        tempx2 = random.randint(5,15)
        tempy1 = random.randint(5,15)
        tempy2 = random.randint(5,15)
        xmin = xmin - tempx1
        xmax = xmax + tempx2
        ymin = ymin - tempy1
        ymax = ymax + tempy2


        for i in range(lenpos):

            '''
            If you need all annotations:
            '''
            mask_image = masks_sorted[i][0]
                                
            cut_mask = cutTheImage(xmin, xmax, ymin, ymax, mask_image)
            cut_mask_normalized = (cut_mask - cut_mask.min()) / (cut_mask.max() - cut_mask.min())  # Normaliza la imagen entre 0 y 1
            cut_mask_uint8 = (cut_mask_normalized * 255).astype(np.uint8)  # Convierte a uint8 (valores entre 0 y 255)
            
            seg.append(cut_mask_uint8)

            cut_img = cutTheImage(xmin, xmax, ymin, ymax, images[i])
            cut_img_normalized = (cut_img - cut_img.min()) / (cut_img.max() - cut_img.min())  # Normaliza la imagen entre 0 y 1
            cut_img_uint8 = (cut_img_normalized * 255).astype(np.uint8)  # Convierte a uint8 (valores entre 0 y 255)

            img.append(cut_img_uint8)

        img = np.array(img)
        seg = np.array(seg)
        reds = np.array(reds)
        blues = np.array(blues)
    else:
        print('No')

    return img, seg #, reds, blues 


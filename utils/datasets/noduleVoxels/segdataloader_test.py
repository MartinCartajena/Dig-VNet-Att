import pandas as pd
import torch.optim as optim
import time
import numpy as np
import os
import torch
from tqdm import tqdm
import math
from bisect import bisect_right
from utils.datasets.noduleVoxels.segdataloader import Dataset 
from utils.datasets.noduleVoxels.segdataloader_lndb import DatasetLNDb 
import csv


def loader(dataset, batch_size, num_workers=4, shuffle=False):
    input_images = dataset

    input_loader = torch.utils.data.DataLoader(dataset=input_images,
                                               batch_size=batch_size,
                                               shuffle=shuffle,
                                               num_workers=num_workers)

    return input_loader

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


def writeTXT(filename, lines):
    with open(filename, 'w') as f:
        for line in lines:
            f.write(line+'\n')

def writeCSV(filename, lines):
    with open(filename, "w", newline='') as f:
        csvwriter = csv.writer(f)
        csvwriter.writerows(lines)

def readCSV(filename):
    lines = []
    with open(filename, "r") as f:
        csvreader = csv.reader(f)
        for line in csvreader:
            lines.append(line)
    return lines

def tryFloat(value):
    try:
        value = float(value)
    except:
        value = value
    
    return value

def getColumn(lines, columnid, elementType=''):
    column = []
    for line in lines:
        try:
            value = line[columnid]
        except:
            continue
            
        if elementType == 'float':
            value = tryFloat(value)

        column.append(value)
    return column

def get_dataset_test_lndb():
    
    path_LNDb = '/app/imagenes/imagesTs/'
    path_segs = '/app/imagenes/labelsTs/'
    path_int = '/app/imagenes/interTs/'
    path_unio = '/app/imagenes/unionTs/'
    
    test_datas = os.listdir(path_LNDb)
    temp_datas = []
    temp_segs = []
    temp_union = []
    temp_inter = []
    for i in test_datas:
        temp_datas.append(path_LNDb + i)
        temp_segs.append(path_segs + i)
        temp_inter.append(path_int + i)
        temp_union.append(path_unio + i)

    batchsize=1
    width=64
    height=64

    temp_data_dir = []
    temp_data_id = []
    flag = False


    """ Dataload y su init en cache"""
    dataset_test = DatasetLNDb(temp_datas, temp_segs, temp_inter, temp_union, width=width, height=height)

    dataloader_test = loader(dataset_test, batchsize, num_workers=0)

    return dataset_test



def get_dataset_test(config):

    """ Datas """    
    test_datas = readCSV(os.path.join('/app/ugs/needed/split_csv/testfold.csv'))

    temp_test_datas = []

    for one in test_datas:
        one_temp = one[0].split('/')[-1]
        one_list = one_temp.split('_')
        temp_test_datas.append(one_list[0] + '_' + one_list[1] + '_' + one_list[2])
        
    batchsize=1
    width=64
    height=64

    """ From nodules to """

    malig =  pd.read_csv(config.csvPath+'malignancy2.csv', header = None)
    dirs = readCSV(config.csvPath + 'directorio_imagenes2.csv')

    mid_files = os.listdir(config.maskPath2)

    temp_data_dir = []
    temp_data_id = []
    flag = False

    for i in range(len(test_datas)):
        A= test_datas[i][0].split('_')
        for j in range(len(malig)):
            id = caseid_to_scanid(malig.loc[j][1])
            if id == A[0] and str(malig.loc[j][2])==A[2] and str(malig.loc[j][3])==A[1]:
                temp_data_id.append(list(malig.loc[j]))
                temp_data_dir.append(config.midPath + '/'.join(dirs[j][0].split('/')[9:]))
                break

    """ Dataload y su init en cache"""
    dataset_test = Dataset(temp_data_dir, temp_data_id, width=width, height=height)

    dataloader_test = loader(dataset_test, batchsize, num_workers=0)
    # for batch_idx, (inputs,  targets_u, targets_i, targets_s) in enumerate(dataloader_test):
    #     print("Cache ON")

    # # dataloader_test.dataset.set_use_cache(True)
    # dataloader_test = loader(dataset_test, batchsize, num_workers=4)

    return dataset_test
    
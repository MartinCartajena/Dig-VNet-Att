import random
import pandas as pd
import csv
from utils.datasets.noduleVoxels.config import Config 
import os

from utils.datasets.noduleVoxels.segdataloader import Dataset 
from utils.datasets.transform import transforms

config = Config()

fold = 1
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

def readCSV(filename):
    lines = []
    with open(filename, "r") as f:
        csvreader = csv.reader(f)
        for line in csvreader:
            lines.append(line[0])
    return lines

malig =  pd.read_csv(config.csvPath+'malignancy2.csv', header = None)

def get_dataset(config, mode='train', batchsize=64, width=64, height=64, depth = 16):

    train_datas = []
    train_masks = []
    for index in config.training_fold_index:
        tempdata = readCSV(os.path.join(config.csvPath, 'datafold' + str(index) + '.csv'))
        tempmask = readCSV(os.path.join(config.csvPath, 'mask_fold' + str(index) + '.csv'))

        train_datas += tempdata
        train_masks += tempmask
    
    test_datas = readCSV(os.path.join(config.csvPath, 'datafold' + str(config.test_fold_index[0]) + '.csv'))

    if mode=='train':
        # remove features labels
        temp_train_datas = []
        for one in train_datas:
            one_temp = one.split('/')[-1]
            one_list = one_temp.split('_')
            temp_train_datas.append(one_list[0] + '_' + one_list[1] + '_' + one_list[2])
        
        temp_test_datas = []
        for one in test_datas:
            one_temp = one.split('/')[-1]
            one_list = one_temp.split('_')
            temp_test_datas.append(one_list[0] + '_' + one_list[1] + '_' + one_list[2])

        mid_files = os.listdir(config.maskPath2)

        dirs = readCSV(config.csvPath + 'directorio_imagenes2.csv')


        temp_data_dir = []
        temp_data_id = []
        flag = False

        for i in range(len(train_datas)):
            A= train_datas[i].split('_')
            for j in range(len(malig)):
                id = caseid_to_scanid(malig.loc[j][1])
                if id == A[0] and str(malig.loc[j][2])==A[2] and str(malig.loc[j][3])==A[1]:
                    temp_data_id.append(list(malig.loc[j]))
                    temp_data_dir.append(config.midPath + '/'.join(dirs[j].split('/')[9:]))
                    break

        test_data_dir = []
        test_data_id = []

        for i in test_datas:
            A= i.split('_')
            for j in dirs:
                B = j.split('/')
                if B[9] == A[0]:
                    test_data_dir.append(config.midPath + '/'.join(B[9:]))
                    idx = dirs.index(j)
                    break
            
            test_data_id.append(list(malig.loc[idx]))


        print('***********')
        print('the length of train data: ', len(temp_data_dir))
        print('the length of test data: ', len(test_data_dir))
        print('-----------')
        
        dataset = Dataset(temp_data_dir, temp_data_id, data_aug=False, transform=None,  width=width, height=height, depth = 16)
        dataset_val = Dataset(test_data_dir, test_data_id, width=width, height=height, depth = 16)
        
        return dataset, dataset_val

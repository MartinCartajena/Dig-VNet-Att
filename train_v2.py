import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluate.loss.dice_loss import DiceLoss
from evaluate.loss.dice_loss import SoftDiceLoss
from evaluate.loss.dice_loss import dsc
from evaluate.loss.dice_loss import soft_dsc

import torchvision.transforms as transforms_tv
import utils.prepare.promise12 as promise12
from utils.prepare.load import load_npy_files_from_directory
import models.VNet_v1 as VNet_v1

from utils.prepare.dig_module import BitwiseImageTransformer


def main(args):
    makedirs(args)
    device = torch.device("cpu" if not torch.cuda.is_available() else args.device)

    loader_train, loader_valid = data_loaders(args)
    loaders = {"train": loader_train, "valid": loader_valid}

    vnet = VNet_v1.VNet()
    vnet.to(device)

    # dsc_loss = DiceLoss()
    softdsc_loss = SoftDiceLoss()
    
    best_validation_dsc = 0.0

    optimizer = optim.Adam(vnet.parameters(), lr= args.lr)
    
    trainF = open(os.path.join("./results/logs/", 'train_images.csv'), 'w')
    validF = open(os.path.join("./results/logs/", 'validation_images.csv'), 'w')

    loss_train = []
    loss_valid = []

    step = 0

    for epoch in range(args.epochs):
        for phase in ["train", "valid"]:
            if phase == "train":
                vnet.train()
            else:
                vnet.eval()

            validation_pred = []
            validation_true = []

            for i, data in enumerate(loaders[phase]):
                

                if phase == "train":
                    step += 1

                x, y_true, id = data
                x, y_true = x.to(device), y_true.to(device)
                
                # dig_module = BitwiseImageTransformer(x)    
                # dig_x = dig_module.transform()

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                
                    y_pred = vnet(x)
                    loss = softdsc_loss(y_pred, y_true)               

                    # y_true = y_true.to(dtype=torch.long)
                    # loss_function = nn.CrossEntropyLoss(reduction='mean')
                    # loss = loss_function(y_pred, y_true)
                    
                    if phase == "valid":
                        loss_valid.append(loss.item())
                        y_pred_np = y_pred.detach().cpu().numpy()
                        validation_pred.extend(
                            [y_pred_np[s] for s in range(y_pred_np.shape[0])]
                        )
                        y_true_np = y_true.detach().cpu().numpy()
                        validation_true.extend(
                            [y_true_np[s] for s in range(y_true_np.shape[0])]
                        )           

                    if phase == "train":
                        loss_train.append(loss.item())
                        loss.backward()
                        optimizer.step()

                if phase == "train" and (step + 1) % 10 == 0:
                    loss_train = []

            if phase == "train":
                trainF.write('{},{}\n'.format(epoch, np.mean(loss_train)))
                trainF.flush()
            
            if phase == "valid":
                validF.write('{},{}\n'.format(epoch, np.mean(loss_valid)))
                validF.flush()
                # mean_dsc = np.mean(
                #     dsc_per_volume(
                #         validation_pred,
                #         validation_true,
                #         loader_valid.dataset.patient_slice_index,
                #     )
                # )
                
                mean_dsc = np.mean(
                    dsc_per_volume_not_flatten(
                        validation_pred,
                        validation_true
                    )
                )
                                
                if mean_dsc > best_validation_dsc:
                    best_validation_dsc = mean_dsc
                    torch.save(vnet.state_dict(), os.path.join(args.weights, "vnet_images.pt"))
                loss_valid = []

    print("Best validation mean DSC: {:4f}".format(best_validation_dsc))
    trainF.close()
    validF.close()


def data_loaders(args):
    
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.device else {}
    
    trainTransform = transforms_tv.Compose([
        transforms_tv.ToTensor()
    ])
    testTransform = transforms_tv.Compose([
        transforms_tv.ToTensor()
    ])
    
    imagesTr_path = os.path.join(args.image_path, "imagesTr")
    numpyImages = load_npy_files_from_directory(imagesTr_path, "imagesTr")
            
    labelsTr_path = os.path.join(args.image_path, "labelsTr")
    numpyGT = load_npy_files_from_directory(labelsTr_path, "labelsTr")

    trainSet = promise12.PROMISE12(mode='train', images=numpyImages, GT=numpyGT, transform=trainTransform, data_format=args.data_format)
    loader_train = DataLoader(trainSet, batch_size=args.batch_size, shuffle=True, **kwargs)


    imagesVal_path = os.path.join(args.image_path, "imagesVal")
    numpyImages = load_npy_files_from_directory(imagesVal_path, "imagesVal")
            
    labelsVal_path = os.path.join(args.image_path, "labelsVal")
    numpyGT = load_npy_files_from_directory(labelsVal_path, "labelsVal")
            
    valSet = promise12.PROMISE12(mode='test', images=numpyImages, GT=numpyGT, transform=testTransform, data_format=args.data_format)
    loader_valid = DataLoader(valSet, batch_size=args.batch_size, shuffle=True, **kwargs)

    return loader_train, loader_valid

# si la prediccion esta flatten
def dsc_per_volume(validation_pred, validation_true, patient_slice_index):
    dsc_list = []
    num_slices = np.bincount([p[0] for p in patient_slice_index])
    index = 0
    for p in range(len(num_slices)):
        y_pred = np.array(validation_pred[index : index + num_slices[p]])
        y_true = np.array(validation_true[index : index + num_slices[p]])
        dsc_list.append(dsc(y_pred, y_true))
        index += num_slices[p]
    return dsc_list

def dsc_per_volume_not_flatten(validation_pred, validation_true):
    dsc_list = []
    for i in range(len(validation_true)):
        y_pred = validation_pred[i].flatten() 
        y_true = validation_true[i].flatten()
        dsc_list.append(soft_dsc(y_pred, y_true))
        
    return dsc_list

def log_loss_summary(logger, loss, step, prefix=""):
    logger.scalar_summary(prefix + "loss", np.mean(loss), step)

def makedirs(args):
    os.makedirs(args.weights, exist_ok=True)
    # os.makedirs(args.logs, exist_ok=True)

def snapshotargs(args):
    args_file = os.path.join(args.logs, "args.json")
    with open(args_file, "w") as fp:
        json.dump(vars(args), fp)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Training vNET model for segmentation of lung CT"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="input batch size for training (default: 16)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="number of epochs to train (default: 100)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        help="initial learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="device for training (default: cuda:0)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="number of workers for data loading (default: 4)",
    )
    parser.add_argument(
        "--vis_images",
        type=int,
        default=200,
        help="number of visualization images to save in log file (default: 200)",
    )
    parser.add_argument(
        "--vis_freq",
        type=int,
        default=10,
        help="frequency of saving images to log file (default: 10)",
    )
    parser.add_argument(
        "--weights", type=str, default="./weights", help="folder to save weights"
    )
    parser.add_argument(
        "--logs", type=str, default="./logs", help="folder to save logs"
    )
    parser.add_argument(
        '--image_path', 
        type=str, 
        default=''
    )
    parser.add_argument(
        '--data_format', 
        type=str, 
        default='mhd'
    )  


    args = parser.parse_args()
    main(args)

import json
import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.prepare.prepareLoss import dsc_per_volume_not_flatten

from evaluate.loss.dice_loss import DiceLoss, SoftDiceLoss, dsc, soft_dsc
import torchvision.transforms as transforms_tv
import utils.prepare.promise12 as promise12
from utils.prepare.load import load_npy_files_from_directory
import models.VNet_v1 as VNet_v1
from utils.prepare.dig_module import BitwiseImageTransformer
from config import get_args  

def main(args):
    makedirs(args)
    device = torch.device("cpu" if not torch.cuda.is_available() else args.device)

    loader_train, loader_valid = data_loaders(args)
    loaders = {"train": loader_train, "valid": loader_valid}

    vnet = VNet_v1.VNet()
    vnet.to(device)

    dsc_loss = DiceLoss()
    softdsc_loss = SoftDiceLoss()

    best_validation_dsc = 0.0

    optimizer = optim.Adam(vnet.parameters(), lr=args.lr)

    trainF = open(os.path.join("./results/logs/", 'train.csv'), 'w')
    validF = open(os.path.join("./results/logs/", 'validation.csv'), 'w')

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

                dig_module = BitwiseImageTransformer(x)    
                dig_x = dig_module.transform()

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):

                    y_pred = vnet(x)
                    loss = softdsc_loss(y_pred, y_true)

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

                mean_dsc = np.mean(
                    dsc_per_volume_not_flatten(
                        validation_pred,
                        validation_true
                    )
                )

                if mean_dsc > best_validation_dsc:
                    best_validation_dsc = mean_dsc
                    torch.save(vnet.state_dict(), os.path.join(args.weights, "vnet.pt"))
                loss_valid = []

    print("Best validation mean DSC: {:4f}".format(best_validation_dsc))
    trainF.close()
    validF.close()


def data_loaders(args):
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.device else {}

    trainTransform = transforms_tv.Compose([transforms_tv.ToTensor()])
    testTransform = transforms_tv.Compose([transforms_tv.ToTensor()])

    imagesTr_path = os.path.join(args.image_path, "imagesTr")
    numpyImages = load_npy_files_from_directory(imagesTr_path)

    labelsTr_path = os.path.join(args.image_path, "labelsTr")
    numpyGT = load_npy_files_from_directory(labelsTr_path)

    trainSet = promise12.PROMISE12(mode='train', images=numpyImages, GT=numpyGT, transform=trainTransform, data_format=args.data_format)
    loader_train = DataLoader(trainSet, batch_size=args.batch_size, shuffle=True, **kwargs)

    imagesVal_path = os.path.join(args.image_path, "imagesVal")
    numpyImages = load_npy_files_from_directory(imagesVal_path)

    labelsVal_path = os.path.join(args.image_path, "labelsVal")
    numpyGT = load_npy_files_from_directory(labelsVal_path)

    valSet = promise12.PROMISE12(mode='test', images=numpyImages, GT=numpyGT, transform=testTransform, data_format=args.data_format)
    loader_valid = DataLoader(valSet, batch_size=args.batch_size, shuffle=True, **kwargs)

    return loader_train, loader_valid


def makedirs(args):
    os.makedirs(args.weights, exist_ok=True)

if __name__ == "__main__":
    args = get_args() 
    main(args)

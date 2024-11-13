import json
import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime

from utils.prepare.prepareLoss import dsc_per_volume_not_flatten

from evaluate.loss.dice_loss import DiceLoss, SoftDiceLoss, dsc, soft_dsc
import torchvision.transforms as transforms_tv
import utils.prepare.promise12 as promise12
from utils.prepare.load import load_npy_files_from_directory
import models.VNet as VNet
from utils.prepare.dig_module import BitwiseImageTransformer
from config import get_args  

from utils.datasets.transform import transforms
from utils.datasets.lungNoduleSegmentationDataset import LungNoduleSegmentationDataset as Dataset


def main(args):
    makedirs(args)
    device = torch.device("cpu" if not torch.cuda.is_available() else args.device)

    loader_train, loader_valid = data_loader(args)
    loaders = {"train": loader_train, "valid": loader_valid}

    vnet = VNet.VNet()
    vnet.to(device)

    # dsc_loss = DiceLoss()
    softdsc_loss = SoftDiceLoss()

    best_validation_dsc = 0.0

    optimizer = optim.Adam(vnet.parameters(), lr=args.lr)
    
    actual_date = datetime.now().strftime("%Y%m%d_%H%M%S")

    trainF = open(os.path.join("./results/logs/", f'train_{actual_date}.csv'), 'w')
    validF = open(os.path.join("./results/logs/", f'validation_{actual_date}.csv'), 'w')

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

                x, y_true = data
                x, y_true = x.to(device), y_true.to(device)

                """ Dig_Sep module: 8 bits module """
                # dig_module = BitwiseImageTransformer(x)    
                # dig_x = dig_module.transform()

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
                    torch.save(vnet.state_dict(), os.path.join(args.weights, f"vnet_{actual_date}.pt"))
                loss_valid = []

    print("Best validation mean DSC: {:4f}".format(best_validation_dsc))
    trainF.close()
    validF.close()


def data_loader(args):
    dataset_train, dataset_valid = datasets(args)

    loader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers
    )
    loader_valid = DataLoader(
        dataset_valid,
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=args.workers
    )

    return loader_train, loader_valid


def datasets(args):
    train = Dataset(
        root_dir=args.data_path, 
        split='train', 
        transform=None # transforms(scale=args.aug_scale, angle=args.aug_angle, flip_prob=0.5),
    )
    valid = Dataset(
        root_dir=args.data_path,
        split='val', 
        transform=None
    )
    return train, valid


def makedirs(args):
    os.makedirs(args.weights, exist_ok=True)

if __name__ == "__main__":
    args = get_args() 
    main(args)

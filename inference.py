import argparse
import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from medpy.filter.binary import largest_connected_component
from skimage.io import imsave
from torch.utils.data import DataLoader
from tqdm import tqdm
import mlflow

from evaluate.loss.dice_loss import soft_dsc
from utils.prepare.dig_module import BitwiseImageTransformer
from utils.prepare.prepareLoss import dsc_per_volume_not_flatten

import models.Dig_Sep_VNet_CBAM as VNet_CBAM
from utils.datasets.train_datasets.lungNodSeg import LungNodSeg as Dataset


def main(args):
    makedirs(args)
    device = torch.device("cpu" if not torch.cuda.is_available() else args.device)

    loader = data_loader(args)

    with torch.set_grad_enabled(False):
        
        if args.weights != None:
        
            vnet = VNet_CBAM.VNet_CBAM(16, args.loss)
            
            state_dict = torch.load(args.weights, map_location=device)
            vnet.load_state_dict(state_dict)
        else:
            vnet = mlflow.pytorch.load_model(args.model_uri)

        vnet.eval()
        vnet.to(device)

        for i, data in tqdm(enumerate(loader)):
            x, y_true, name = data
            x, y_true = x.to(device), y_true.to(device)

            dig_module = BitwiseImageTransformer(x)    
            dig_x = dig_module.transform()
                    
            y_pred = vnet(dig_x)
            
            y_pred_classes = torch.argmax(y_pred, dim=1)  # Shape: [batch_size, depth, height, width]
            
            y_pred_np = y_pred_classes.cpu().numpy()

            for j in range(y_pred_np.shape[0]):  # Iterate over batch
                new_name = str(name[j]).replace(".npy", "")
                file_name = f"{new_name}_pred.npy" 
                save_path = os.path.join(args.output_dir, file_name)
                np.save(save_path, y_pred_np[j])  # Save individual prediction

                print(f"Saved prediction: {save_path}")
                

def data_loader(args):
    dataset = Dataset(
        root_dir=args.images,
        split='test', 
        transform=None
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, drop_last=False, num_workers=1
    )
    return loader


def makedirs(args):
    os.makedirs(args.output_dir, exist_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inference for segmentation of brain MRI"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="device for training (default: cuda:0)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="input batch size for training (default: 32)",
    )
    parser.add_argument(
        "--weights", type=str, default=None, help="path to weights file"
    )
    parser.add_argument(
        "--model_uri", type=str, help="mlflow model path"
    )
    parser.add_argument(
        "--images", type=str, required=True,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results/predicts",
        help="folder for saving images with prediction outlines",
    )
    parser.add_argument(
        "--figure",
        type=str,
        default="./dsc.png",
        help="filename for DSC distribution figure",
    )
    parser.add_argument(
        "--loss",
        type=str,
    )

    args = parser.parse_args()
    main(args)
import argparse
import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import mlflow

from utils.prepare.dig_module import BitwiseImageTransformer

import models.Dig_Sep_VNet_CBAM as VNet_CBAM
from utils.datasets.train_datasets.lungNoduleSegmentationDataset import LungNodSeg as Dataset

from utils.datasets.noduleVoxels.segdataloader import loader as loader_s
from utils.datasets.noduleVoxels.config import Config

from utils.datasets.noduleVoxels.segdataloader_test import get_dataset_test
from utils.datasets.noduleVoxels.segdataloader_test import get_dataset_test_lndb


def main(args):
    makedirs(args)
    device = torch.device("cpu" if not torch.cuda.is_available() else args.device)

    if args.dataset == '1':
        loader = data_loader(args)
    
    elif args.dataset == '2':
        config = Config()
        
        # dataset = get_dataset_test(config)   # 64
        dataset = get_dataset_test_lndb()   # 64

        
        """ init cache in dataset """
        # dataloader = loader(dataset, args.batch_size, num_workers=0)
        # dataloader_val = loader(dataset_val, args.batch_size, num_workers=0)
        # for idx, data in enumerate(dataloader):
        #     print("Cache ON: train", idx)

        # for idx, data in enumerate(dataloader_val):
        #     print("Cache ON: val", idx)
            
        # dataset.set_use_cache(True)
        # dataset_val.set_use_cache(True)

        """ fin ini cache"""

        loader = loader_s(dataset, args.batch_size, num_workers=8)

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
            x, name, y_true = data
            x = x.to(device)
            
            x = torch.squeeze(x, dim=1)
            
            if y_true != []:
                y_true = torch.squeeze(y_true, dim=1)

            dig_module = BitwiseImageTransformer(x)    
            dig_x = dig_module.transform()
                    
            y_pred = vnet(dig_x)
            
            y_pred_classes = torch.argmax(y_pred, dim=1)  # Shape: [batch_size, depth, height, width]
            
            y_pred_np = y_pred_classes.detach().cpu().numpy()
            
            for i in range(y_pred_np.shape[0]):
                if len(np.unique(y_pred_np[i])) > 2:
                    print("Raro....")
                    
            for j in range(y_pred_np.shape[0]):  # Iterate over batch
                new_name = str(name[j]).replace(".npy", "")
                file_name = f"{new_name}_pred.npy" 
                mask_name = f"{new_name}.npy"
                save_path = os.path.join(args.output_dir, file_name)
                save_path_mask =  os.path.join("/app/results/preds/LNDb_a/mask/", mask_name)
                
                i = 1
                while os.path.exists(save_path):
                    if not save_path.endswith(f"_{i}_pred.npy"):
                        save_path = save_path.replace("_pred.npy", f"_{i}_pred.npy")
                        save_path_mask = save_path_mask.replace(".npy", f"_{i}.npy")
                    else:
                        save_path = save_path.replace(f"_{i}_pred.npy", f"_{i+1}_pred.npy")
                        save_path_mask = save_path_mask.replace(f"_{i}.npy", f"_{i+1}.npy")
                
                    i += 1
                    
                np.save(save_path, y_pred_np[j])  # Save individual prediction
                if y_true != []:
                    np.save(save_path_mask, y_true[j]) 
                    
                print(f"Saved prediction: {save_path}")
                

def data_loader(args):
    dataset = Dataset(
        root_dir=args.images,
        split='test', 
        transform=None
    )
    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        drop_last=False, 
        num_workers=1
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
    parser.add_argument(
        "--dataset",
        type=str,
    )
    
    args = parser.parse_args()
    main(args)
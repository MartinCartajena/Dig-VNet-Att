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
import models.VNet as VNet
import models.Dig_Sep_VNet_CBAM as VNet_CBAM
from utils.prepare.dig_module import BitwiseImageTransformer
from config import get_args  

import mlflow
import mlflow.pytorch

from utils.datasets.transform import transforms
from utils.datasets.lungNoduleSegmentationDataset import LungNoduleSegmentationDataset as Dataset
from utils.prepare.data_augmentation import DataAugmentation


def main(args):
    makedirs(args)
    device = torch.device("cpu" if not torch.cuda.is_available() else args.device)
    
    actual_date = datetime.now().strftime("%Y%m%d_%H%M%S") # TODO: horario aleman... 

    experiment_name = f"Experiment_Segmentation_{actual_date}"
    mlflow.set_experiment(experiment_name)

    dataset_train, dataset_valid = datasets(args)

    """ INIT preprcesado en cache y aumento de datos en cache """
    if args.preprocess:
        
        try:
            loader_train, loader_val = data_loader(args, dataset_train, dataset_valid, preprocess=True)

            for idx, data in enumerate(loader_train):
                print("Preprocess train:", idx)

            for idx, data in enumerate(loader_val):
                print("Preprocess val:", idx)
                
            dataset_train.setCache(True)
            dataset_valid.setCache(True)
        except Exception as e:
            print(f"Error en DataLoader: {e}")

    """ FIN preprcesado y aumento de datos en cache """

    loader_train, loader_val = data_loader(args, dataset_train, dataset_valid)
    loaders = {"train": loader_train, "valid": loader_val}
    

    if args.dig_sep:
        vnet = VNet_CBAM.VNet_CBAM(16, args.loss)
    else:
        vnet = VNet_CBAM.VNet_CBAM(1, args.loss)

    vnet.to(device)

    if args.loss == "softdice":
        loss_function = SoftDiceLoss()
    elif args.loss == "crossentropy":
        loss_function = torch.nn.CrossEntropyLoss()
    elif args.loss == "dice":
        # TODO: si encuentras alguna manera de no romper el grafo de grands de torch para el backpropagation
        print(f"Dice no esta hecho todavia por problemas de diferenciabilidad...")
    else:
        raise ValueError("Invalid loss type. Choose 'softdice' or 'crossentropy'.")

    optimizer = optim.Adam(vnet.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', # decir que valor es el mejor, en este caso min porque es loss 
        factor=0.1, # multiplicador
        patience=10, # num de epocas hasta que cambia el lr
        verbose=True # imprimir mensajes cuando lr cambia, aun y todo lo meto en mlflow
    )

    """ Early stopping """
    early_stopping_patience = 30 # args.patience  # 10
    epochs_no_improve = 0
    best_validation_loss = float("inf")

    loss_train = []

    with mlflow.start_run():
        
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("learning_rate", args.lr)
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("dig_sep", args.dig_sep)
        mlflow.log_param("loss_function", args.loss)

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

                    if args.dig_sep:
                        """ Dig_Sep module: 8 bits module """
                        dig_module = BitwiseImageTransformer(x)    
                        dig_x = dig_module.transform()

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == "train"):

                        x = x.unsqueeze(1) # convert to [16, 1, 16, 96, 96]
                        
                        if args.dig_sep:
                            y_pred = vnet(dig_x)
                        else:
                            y_pred = vnet(x)
                            
                        if phase == "train":
                            if args.loss == "crossentropy":
                                y_true_adjusted = y_true.long()
                                loss = loss_function(y_pred, y_true_adjusted)
                            else:
                                loss = loss_function(y_pred, y_true)


                        if phase == "valid":
                            if args.loss == "crossentropy":
                                y_pred_np = y_pred.detach()
                                y_true_np = y_true.detach()
                            else:
                                y_pred_np = y_pred.detach().cpu().numpy()
                                y_true_np = y_true.detach().cpu().numpy()

                            validation_pred.extend(
                                [y_pred_np[s] for s in range(y_pred_np.shape[0])]
                            )
                            validation_true.extend(
                                [y_true_np[s] for s in range(y_true_np.shape[0])]
                            )

                        if phase == "train":
                            loss_train.append(loss.item())
                            loss.backward()
                            optimizer.step()

                if phase == "train":
                    epoch_loss = np.round(np.mean(loss_train), 6)
                    print(f"Train: Epoch {epoch} --> {args.loss} loss {epoch_loss}")
                    mlflow.log_metric(f"Train_{args.loss}_loss", epoch_loss, step=epoch)
                    loss_train = []             

                if phase == "valid":  
                                                    
                    if args.loss == "crossentropy":
                        validation_loss = []
                        for vp, vt in zip(validation_pred, validation_true):
                            
                            vt_tensor = vt.unsqueeze(0).long()
                            vp_tensor = vp.unsqueeze(0)
                            
                            loss_value = loss_function(vp_tensor, vt_tensor).item()

                            validation_loss.append(loss_value)
                    
                        current_loss = np.mean(validation_loss)
                        
                        mean_softdsc = np.mean(
                            dsc_per_volume_not_flatten(
                                validation_pred,
                                validation_true
                            )
                        )
                        
                        mlflow.log_metric("Softdice", mean_softdsc, step=epoch)


                    else:
                        mean_softdsc = np.mean(
                            dsc_per_volume_not_flatten(
                                validation_pred,
                                validation_true
                            )
                        )
                        current_loss = 1 - mean_softdsc
                        
                        mlflow.log_metric("validation_softdsc", mean_softdsc, step=epoch)


                    print(f"Validation: Epoch {epoch} --> {args.loss} loss {current_loss}")
                    mlflow.log_metric(f"Validation_{args.loss}_loss", current_loss, step=epoch)

                    scheduler.step(current_loss)
                    current_lr = optimizer.param_groups[0]['lr']
                    mlflow.log_metric("learning_rate", current_lr, step=epoch)

                    if current_loss < best_validation_loss:
                        best_validation_loss = current_loss
                        """ Seguire guardando el modelo asi, aunque tambien lo guarde en mlflow"""
                        torch.save(vnet.state_dict(), os.path.join(args.weights, args.weights_name))
                        mlflow.pytorch.log_model(vnet, "model")  # Guarda el modelo en MLflow                
                    
                        epochs_no_improve = 0  # Early stopping basado en el loss
                    else:
                        epochs_no_improve += 1

                    if epochs_no_improve >= early_stopping_patience:
                        print(f"Early stopping triggered at epoch {epoch}")
                        break
                                   
            # Break externo si se activa el early stopping
            if epochs_no_improve >= early_stopping_patience:
                break
            
        print("Best validation loss: {:4f}".format(best_validation_loss))
        mlflow.log_metric(f"best_validation_{args.loss}_loss", best_validation_loss) 



########################################################################## DATASET & DATALOADER ##########################################################################

def data_loader(args, dataset_train, dataset_valid, preprocess=False):

    if preprocess:
        loader_train = DataLoader(
            dataset_train,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0
        )
        
        loader_valid = DataLoader(
            dataset_valid,
            batch_size=args.batch_size,
            drop_last=False,
            num_workers=0
        )
    else:
        loader_train = DataLoader(
            dataset_train,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
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
        preprocess=args.preprocess,
        # transform= transforms(
        #                 angle=30,
        #                 horizontal_flip_prob=0.5,
        #                 vertical_flip_prob=0.5,
        #                 salt_pepper_prob={'prob': 0.25, 'amount': 0.05, 'salt_ratio': 0.5}
        #             )
        transform=None,
        data_aug=DataAugmentation(noise_amount=0.03, salt_ratio=0.5)
    )
    
    valid = Dataset(
        root_dir=args.data_path,
        split='val', 
        preprocess=args.preprocess,
        transform=None
    )
    return train, valid


########################################################################## INIT ##########################################################################


def makedirs(args):
    os.makedirs(args.weights, exist_ok=True)

if __name__ == "__main__":
    args = get_args() 
    main(args)

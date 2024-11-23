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


def main(args):
    makedirs(args)
    device = torch.device("cpu" if not torch.cuda.is_available() else args.device)
    
    
    actual_date = datetime.now().strftime("%Y%m%d_%H%M%S") # TODO: horario aleman... 

    experiment_name = f"Experiment_Segmentation_{actual_date}"
    mlflow.set_experiment(experiment_name)


    loader_train, loader_valid = data_loader(args)
    loaders = {"train": loader_train, "valid": loader_valid}

    if args.dig_sep:
        vnet = VNet_CBAM.VNet_CBAM(16)
    else:
        vnet = VNet_CBAM.VNet_CBAM(1)

    vnet.to(device)

    if args.loss == "softdice":
        loss_function = SoftDiceLoss()
    elif args.loss == "crossentropy":
        loss_function = torch.nn.CrossEntropyLoss()
    elif args.loss == "dice":
        print(f"Dice no esta hecho todavia por problemas de diferenciabilidad...")
    else:
        raise ValueError("Invalid loss type. Choose 'softdice' or 'crossentropy'.")


    optimizer = optim.Adam(vnet.parameters(), lr=args.lr)

    # Inicialización del scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', # decir que valor es el mejor, en este caso min porque es loss 
        factor=0.1, # multiplicador
        patience=10, # num de epocas hasta que cambia el lr
        verbose=True # imprimir mensajes cuando lr cambia, aun y todo lo meto en mlflow
    )

    """ Para quitar: Forma antigua de llevar el loss """
    # trainF = open(os.path.join("./results/logs/", f'train_{actual_date}.csv'), 'w')
    # validF = open(os.path.join("./results/logs/", f'validation_{actual_date}.csv'), 'w')
    
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
                                # Convertir y_pred para que tenga logits de dos clases. Esto convierte y_pred a un tensor de shape: [batch_size, 2, 16, 96, 96]
                                y_pred_adjusted = torch.cat([(1 - y_pred).unsqueeze(1), y_pred.unsqueeze(1)], dim=1)

                                # Asegurarse de que y_true sea de tipo long
                                y_true_adjusted = y_true.squeeze(1).long()
                                
                                loss = loss_function(y_pred_adjusted, y_true_adjusted)
                            else:
                                loss = loss_function(y_pred, y_true)


                        if phase == "valid":
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

                if phase == "train":
                    epoch_loss = np.round(np.mean(loss_train), 6)
                    print(f"Train: Epoch {epoch} --> {args.loss} loss {epoch_loss}")
                    mlflow.log_metric(f"Train_{args.loss}_loss", epoch_loss, step=epoch)
                    
                    """ Para quitar: Forma antigua de llevar el loss """
                    # trainF.write('{},{}\n'.format(epoch, np.round(np.mean(loss_train), 6)))
                    # trainF.flush()

                if phase == "valid":  
                                
                    mean_softdsc = np.mean(
                        dsc_per_volume_not_flatten(
                            validation_pred,
                            validation_true
                        )
                    )
                    print(f"Validation: Epoch {epoch} --> Softdice loss {1-mean_softdsc}")

                    
                    if args.loss == "crossentropy":
                        validation_loss = []
                        for vp, vt in zip(validation_pred, validation_true):
                            vp_tensor = torch.tensor(vp, device=device)
                            vt_tensor = torch.tensor(vt, device=device)

                            vt_tensor = vt_tensor.unsqueeze(0).long()
                            vp_tensor_adjusted = torch.stack([1 - vp_tensor, vp_tensor], dim=0).unsqueeze(0)
                            
                            print(f"vt_tensor.min(): {vt_tensor.min()}, vt_tensor.max(): {vt_tensor.max()}")
                            print(f"vp_tensor_adjusted.shape: {vp_tensor_adjusted.shape}, vt_tensor.shape: {vt_tensor.shape}")
                            
                            loss_value = loss_function(vp_tensor_adjusted, vt_tensor).item()

                            validation_loss.append(loss_value)
                    
                        current_loss = np.mean(validation_loss)

                    else:
                        current_loss = 1 - mean_softdsc

                    print(f"Validation: Epoch {epoch} --> {args.loss} loss {current_loss}")
                    mlflow.log_metric(f"Validation{args.loss}_loss", mean_softdsc, step=epoch)
                    mlflow.log_metric("validation_softdsc", mean_softdsc, step=epoch)

                    scheduler.step(current_loss)
                    current_lr = optimizer.param_groups[0]['lr']
                    mlflow.log_metric("learning_rate", current_lr, step=epoch)

                    if current_loss < best_validation_loss:
                        best_validation_loss = current_loss
                        """ Seguire guardando el modelo asi, aunque tambien lo guarde en mlflow"""
                        torch.save(vnet.state_dict(), os.path.join(args.weights, f"model_{actual_date}.pt"))
                        mlflow.pytorch.log_model(vnet, "model")  # Guarda el modelo en MLflow
                    
                    """ Para quitar: Forma antigua de llevar el loss """
                    # validF.write('{},{}\n'.format(epoch, np.round(current_loss, 6)))
                    # validF.flush()
                    
                    # Early stopping basado en el loss
                    if current_loss < best_validation_loss:
                        best_validation_loss = current_loss
                        epochs_no_improve = 0
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

        """ Para quitar: Forma antigua de llevar el loss """
        # trainF.close()
        # validF.close()


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
        transform=None # transforms( angle=args.aug_angle, flip_prob=0.5), # scale=args.aug_scale
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

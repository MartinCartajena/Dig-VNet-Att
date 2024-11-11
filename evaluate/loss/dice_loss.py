import torch.nn as nn
import numpy as np
from medpy.filter.binary import largest_connected_component


class DiceLoss(nn.Module):

    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1.0

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        dsc = (2. * intersection + self.smooth) / (
            y_pred.sum() + y_true.sum() + self.smooth
        )
        return 1. - dsc
    

class SoftDiceLoss(nn.Module):
    
    def __init__(self, smooth=1.0):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size(), "Las dimensiones de predicción y verdad deben coincidir."
        y_pred = y_pred.contiguous().view(-1)
        y_true = y_true.contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        denominator = y_pred.sum() + y_true.sum()
        soft_dice = (2. * intersection + self.smooth) / (denominator + self.smooth)   
        return 1. - soft_dice

    
def dsc(y_pred, y_true, lcc=True):
    if lcc and np.any(y_pred):
        y_pred = np.round(y_pred).astype(int)
        y_true = np.round(y_true).astype(int)
        y_pred = largest_connected_component(y_pred)
    return np.sum(y_pred[y_true == 1]) * 2.0 / (np.sum(y_pred) + np.sum(y_true))


def soft_dsc(y_pred, y_true, lcc=True):
    if lcc and np.any(y_pred):
        y_pred_bin = np.round(y_pred).astype(int)  
        y_true_bin = np.round(y_true).astype(int)
        
        if np.any(y_pred_bin):  
            y_pred_bin = largest_connected_component(y_pred_bin)

        y_pred = y_pred_bin.astype(float)
    
    intersection = np.sum(y_pred * y_true)
    denominator = np.sum(y_pred) + np.sum(y_true)
    
    smooth = 1e-6
    soft_dice = (2. * intersection + smooth) / (denominator + smooth)
    
    return soft_dice

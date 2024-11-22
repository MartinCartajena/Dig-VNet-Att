
from evaluate.loss.dice_loss import soft_dsc

def dsc_per_volume_not_flatten(validation_pred, validation_true):
    dsc_list = []
    for i in range(len(validation_true)):
        y_pred = validation_pred[i].flatten() 
        y_true = validation_true[i].flatten()
        dsc_list.append(soft_dsc(y_pred, y_true))

    return dsc_list
import torch
import torch.nn as nn

class CrossEntropyLoss(nn.Module):
    """
    Implementación de la función de pérdida de entropía cruzada (Cross-Entropy Loss).
    """

    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, input, target):
        """
        Calcula la pérdida de entropía cruzada.

        Parámetros:
        - input: tensor de tamaño (N, C), donde cada elemento representa la probabilidad predicha para cada clase.
        - target: tensor de tamaño (N, C), donde cada elemento representa la etiqueta verdadera en formato one-hot (0 o 1).

        Retorno:
        - Pérdida promedio de entropía cruzada.
        """
        
        _, result_ = input.max(1)
        result_ = torch.squeeze(result_)
        
        if input.is_cuda:
            result = torch.cuda.FloatTensor(result_.size())
            target = torch.cuda.FloatTensor(target.size())
        else:
            result = torch.FloatTensor(result_.size())
            target = torch.FloatTensor(target.size())
            
        result = torch.clamp(result, 1e-12, 1.0)
        
        cross_entropy = torch.mean(torch.sum(target * torch.log(result), dim=0))
        return cross_entropy / 100


def cross_entropy_loss(input, target):
    criterion = CrossEntropyLoss()
    return criterion(input, target)

import pandas as pd
from dataprocess.dataprocess import *
import loss as losses
from metrics import *
import torch.optim as optim
import time
import numpy as np
import os
import torch
from config import Config
import shutil
from tqdm import tqdm
import imageio
import math
from bisect import bisect_right
from model.MyModel import Mymodel
from loss.BCE import BCE_loss

from dataprocess import get_dataset
from segdataloader import loader

config = Config()
if torch.cuda.is_available():
    print('disponible')
else:
    print('cpu')

torch.cuda.set_device(config.gpu)  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = config.arch
if not os.path.isdir('result'):
    os.mkdir('result')
if config.resume is False:
    with open('/app/ugs/result/' + str(os.path.basename(__file__).split('.')[0]) + '_' + model_name + '.txt', 'a+') as f:
        f.seek(0)
        f.truncate()
model = Mymodel(img_ch=1).to(device)
best_dice = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, betas=(0.5, 0.999))    


dataset, dataset_val = get_dataset(config, batchsize=config.batch_size)   # 64


""" init cache in dataset """
# dataloader = loader(dataset, config.batch_size, num_workers=0)
# dataloader_val = loader(dataset_val, config.batch_size, num_workers=0)
# for batch_idx, (inputs,  targets_u, targets_i, targets_s) in enumerate(dataloader):
#     print("Cache ON: train")

# for batch_idx, (inputs,  targets_u, targets_i, targets_s) in enumerate(dataloader_val):
#     print("Cache ON: val")

""" fin ini cache"""

""" re-crear una clase de dalaloader con el mismo dataset que tiene cache"""
dataloader = loader(dataset, config.batch_size, num_workers=4)
dataloader_val = loader(dataset_val, config.batch_size, num_workers=4)

#criterion = losses.BCE

if config.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    if config.evaluate:
        checkpoint = torch.load('/app/checkpoint/' + str(model_name) + '_best.pth.tar')
    else:
        checkpoint = torch.load('/app/checkpoint/' + str(model_name) + '.pth.tar')
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    best_dice = checkpoint['dice']
    start_epoch = config.initepochs

def adjust_lr(optimizer, epoch, eta_max=0.0001, eta_min=0.):
    cur_lr = 0.
    if config.lr_type == 'SGDR':
        i = int(math.log2(epoch / config.sgdr_t + 1))
        T_cur = epoch - config.sgdr_t * (2 ** (i) - 1)
        T_i = (config.sgdr_t * 2 ** i)

        cur_lr = eta_min + 0.5 * (eta_max - eta_min) * (1 + np.cos(np.pi * T_cur / T_i))

    elif config.lr_type == 'multistep':
        cur_lr = config.learning_rate * 0.1 ** bisect_right(config.milestones, epoch)

    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr
    return cur_lr

def dice_score(pred, gt):
    """
    :return save img' dice value in IoUs
    """
    # dices = []
    pred[pred < 0.5] = 0
    pred[pred >= 0.5] = 1
    gt[gt < 0.5] = 0
    gt[gt >= 0.5] = 1
    pred = pred.type(torch.LongTensor)
    pred_np = pred.data.cpu().numpy()
    gt = gt.data.cpu().numpy()
    if not float(np.sum(pred_np) + np.sum(gt)) == 0:
        dice = np.sum(pred_np[gt == 1]) * 2 / float(np.sum(pred_np) + np.sum(gt))
    else:
        print('out')
        dice = None
    return dice

def train(epoch):
    model.train()
    train_loss = 0

    start_time = time.time()
    lr = adjust_lr(optimizer, epoch)

    #for batch_idx, (inputs, lungs, medias, targets_u, targets_i, targets_s) in enumerate(dataloader):
    for batch_idx, (inputs,  targets_u, targets_i, targets_s) in enumerate(dataloader):
        iter_start_time = time.time()
        inputs = inputs.to(device)
        # inputs = inputs.cpu()
        #lungs = lungs.cuda()
        #medias = medias.cuda()
        targets_i = targets_i.to(device)
        targets_u = targets_u.to(device)
        targets_s = targets_s.to(device)
        # targets_i = targets_i.cpu()
        # targets_u = targets_u.cpu()
        # targets_s = targets_s.cpu()

        outputs = model(inputs)
        
        outputs_i_sig = torch.sigmoid(outputs[0])
        outputs_u_sig = torch.sigmoid(outputs[1])
        outputs_s_sig = torch.sigmoid(outputs[2])
        outputs_final_sig = torch.sigmoid(outputs[3])
        
        loss_seg_i = BCE_loss(outputs_i_sig, targets_i)
        loss_seg_u = BCE_loss(outputs_u_sig, targets_u)
        loss_seg_s = BCE_loss(outputs_s_sig, targets_s)
        loss_seg_final = BCE_loss(outputs_final_sig, targets_s)

        loss_all = config.weight_seg1 * loss_seg_final + config.weight_seg2 * (loss_seg_i + loss_seg_u + loss_seg_s)

        optimizer.zero_grad()
        loss_all.backward()
        optimizer.step()

        train_loss += loss_all.item()

        print('Epoch:{}\t batch_idx:{}/All_batch:{}\t duration:{:.3f}\t loss_all:{:.3f}'
          .format(epoch, batch_idx, len(dataloader), time.time()-iter_start_time, loss_all.item()))
        iter_start_time = time.time()
    print('Epoch:{0}\t duration:{1:.3f}\ttrain_loss:{2:.6f}'.format(epoch, time.time()-start_time, train_loss/len(dataloader)))
    
    with open('/app/resultado/' + str(os.path.basename(__file__).split('.')[0]) + '_' + model_name + '.txt', 'a+') as f:
        f.write('Epoch:{0}\t duration:{1:.3f}\t learning_rate:{2:.6f}\t train_loss:{3:.4f}'
          .format(epoch, time.time()-start_time, lr, train_loss/len(dataloader)))

def test(epoch, dices_cortes, idx_cortes):
    global best_dice
    model.eval()
    dices_all_i = []
    dices_all_u = []
    dices_all_s = []
    ious_all_i = []
    ious_all_u = []
    ious_all_s = []
    nsds_all_i = []
    nsds_all_u = []
    nsds_all_s = []
    with torch.no_grad():
        #for batch_idx, (inputs, lungs, medias, targets_u, targets_i, targets_s) in enumerate(dataloader_val):
        for batch_idx, (inputs, targets_u, targets_i, targets_s) in enumerate(dataloader_val):
            inputs = inputs.to(device)

            targets_i = targets_i.to(device)
            targets_u = targets_u.to(device)
            targets_s = targets_s.to(device)


            outputs = model(inputs)

            outputs_final_sig = torch.sigmoid(outputs[3])



            dices_all_i = meandice(outputs_final_sig, targets_i, dices_all_i)
            dices_all_u = meandice(outputs_final_sig, targets_u, dices_all_u)
            dices_all_s = meandice(outputs_final_sig, targets_s, dices_all_s)
            #ious_all_i = meandIoU(outputs_final_sig, targets_i, ious_all_i)
            # ious_all_u = meandIoU(outputs_final_sig, targets_u, ious_all_u)
            # ious_all_s = meandIoU(outputs_final_sig, targets_s, ious_all_s)
            # nsds_all_i = meanNSD(outputs_final_sig, targets_i, nsds_all_i)
            # nsds_all_u = meanNSD(outputs_final_sig, targets_u, nsds_all_u)
            # nsds_all_s = meanNSD(outputs_final_sig, targets_s, nsds_all_s)
            
   
            print('Epoch:{}\tbatch_idx:{}/All_batch:{}\tdice_i:{:.4f}\tdice_u:{:.4f}\tdice_s:{:.4f}\tiou_i:{:.4f}\tiou_u:{:.4f}\tiou_s:{:.4f}'
            .format(epoch, batch_idx, len(dataloader_val), np.mean(np.array(dices_all_i)), np.mean(np.array(dices_all_u)), np.mean(np.array(dices_all_s)), np.mean(np.array(ious_all_i)), np.mean(np.array(ious_all_u)), np.mean(np.array(ious_all_s))))
        with open('/app/resultado/' + str(os.path.basename(__file__).split('.')[0]) + '_' + model_name + '.txt', 'a+') as f:
            f.write('\tdice_i:{:.4f}\tdice_u:{:.4f}\tdice_s:{:.4f}\tdice_std{:.4f}\tiou_i:{:.4f}\tiou_u:{:.4f}\tiou_s:{:.4f}'.format(np.mean(np.array(dices_all_i)), np.mean(np.array(dices_all_u)), np.mean(np.array(dices_all_s)), np.std(np.array(dices_all_s)), np.mean(np.array(ious_all_i)), np.mean(np.array(ious_all_u)), np.mean(np.array(ious_all_s)))+'\n')

    # Save checkpoint.
    if config.resume is False:
        dice = np.mean(np.array(dices_all_s))
        dice_std = np.std(np.array(dices_all_s))
        print('Test accuracy: ', dice)
        print('Test std: ', dice_std)
        state = {
            'model': model.state_dict(),
            'dice': dice,
            'epoch': epoch,
            'optimizer': optimizer.state_dict()
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, '/app/checkpoint/'+str(model_name)+'.pth.tar')

        is_best = False
        if best_dice < dice:
            best_dice = dice
            is_best = True

        if is_best:
            shutil.copyfile('/app/checkpoint/todo_corte_sui_' + str(model_name) + '.pth.tar',
                            '/app/checkpoint/todo_corte_sui_' + str(model_name) + '_best.pth.tar')
        print('Save Successfully')
        print('------------------------------------------------------------------------')
    return dices_cortes, idx_cortes

if __name__ == '__main__':
    dice_cortes = []
    idx_cortes = []

    if config.resume:
        test(start_epoch)
    else:
      
        for epoch in tqdm(range(start_epoch, config.epochs)):
            train(epoch)
            dice_cortes, idx_cortes = test(epoch, dice_cortes, idx_cortes)
            print('Primer epoch')

        nodulos = readCSV(os.path.join(config.csvPath, 'datafold' + str(config.test_fold_index[0]) + '.csv'))
        dice_nodulo= []
        idx_nodulo = []
        df=pd.DataFrame({
            'nombre_corte': idx_cortes,
            'dice_score': dice_cortes
        })
        nombre_nodulo = []
        for i in df.nombre_corte:
            mid = i.split('_')
            nodulo = mid[0] + '_' + mid[1] + '_' + mid[2]
            nombre_nodulo.append(nodulo)
        df['nodulo'] = nombre_nodulo
        df.to_csv('/app/checkpoint/validation_dice_cortes.csv')
        media_nodulo = []
        idx_media = []

        for i in nodulos:
            if i in df.nodulo:
                media = np.mean(df.dice_score[df.nodulo == i])
                media_nodulo.append(media)
                idx_media.append(i)
            

        for cortes, dice in zip(idx_cortes, dice_cortes):
            nodulo_separado = cortes.split('_')
            nodulo = nodulo_separado[0:3]

            

    

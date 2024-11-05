#!/usr/bin/env python3
import os
import sys
import math
import pdb

# from local import *
import time
import torch

import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

import torchvision.transforms as transforms

import torchbiomed.transforms as biotransforms
import torchbiomed.loss as bioloss
import torchbiomed.utils as utils

import SimpleITK as sitk

import shutil

import setproctitle

import models.VNet_v2 as VNet_v2
import models.VNet_v1 as VNet_v1
import utils.DataManager as DM
import utils.prepare.promise12 as promise12
import evaluate.make_graph as make_graph
from functools import reduce
import operator
import evaluate.loss.cross_entropy as ce

from utils.prepare.load import load_npy_files_from_directory

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv3d') != -1:
        nn.init.kaiming_normal_(m.weight)
        m.bias.data.zero_()

def datestr():
    now = time.gmtime()
    return '{}{:02}{:02}_{:02}{:02}'.format(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)


def save_checkpoint(state, path, prefix, filename='checkpoint.pth.tar'):
    prefix_save = os.path.join(path, prefix)
    name = prefix_save + '_' + filename
    torch.save(state, name)


def inference(params, args, loader, model, resultDir):
    
    if args.infer_data_path:
        src = os.path.join(args.infer_data_path, "imagesTs")
    else:
        src = params['ModelParams']['dirInfer']
        
    dst = params['ModelParams']['dirResult']

    model.eval()
    # assume single GPU / batch size 1
    for batch_idx, data in enumerate(loader):
        data, ids = data
          
        # pdb.set_trace()
        _, _, z, y, x = data.shape # need to subset shape of 3-d. by Chao.
        # convert names to batch tensor
        if args.cuda:
            data.pin_memory()
            data = data.cuda()
        with torch.no_grad():
            data = Variable(data)
        output = model(data)
        
        if not os.path.exists(resultDir + "/predict/"):
            os.makedirs(resultDir + "/predict/")
            
        for i in range(output.size(0)):
            _, output_i = output[i,:,:].max(1)
            output_i = output_i.view((x, y, z))
            # pdb.set_trace()
            output_i = output_i.cpu()

            print("save {}".format(ids[i]))
        
            if args.data_format == "mhd":
                itk_img = sitk.ReadImage(os.path.join(src, ids[i]))
                origin = np.array(list(reversed(itk_img.GetOrigin())))
                spacing = np.array(list(reversed(itk_img.GetSpacing())))
                utils.save_updated_image(output_i, resultDir + "/predict/" +  ids[i] + "_predicted.mhd", origin, spacing)
            elif args.data_format == "npy":
                np_output = output_i.numpy()
                np.save(resultDir + "/predict/" + ids[i], np_output)
                
# performing post-train test:
# train.py --resume <model checkpoint> --i <input directory (*.mhd)> --save <output directory>

def noop(x):
    return x

def main(params, args):
    best_prec1 = 100. # accuracy? by Chao
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    resultDir = 'results/vnet.base.{}'.format(datestr())
    nll = True
    if args.dice:
        nll = False
    weight_decay = args.weight_decay
    setproctitle.setproctitle(resultDir)

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    print("build vnet")
    # model = VNet_v2.VNet(elu=False, nll=nll)
    model = VNet_v1.VNet()
    
    batch_size = args.batchSz
    
    if args.cuda:
        torch.cuda.set_device(0) # why do I have to add this line? It seems the below line is useless to apply GPU devices. By Chao.
        model = nn.parallel.DataParallel(model, device_ids=[0])

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        model.apply(weights_init)

    if nll:
        print()
        # train = train_nll
        # test = test_nll
    else:
        train = train_dice
        test = test_dice

    print('  + Number of params: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))
    if args.cuda:
        model = model.cuda()

    if os.path.exists(resultDir):
        shutil.rmtree(resultDir)
    os.makedirs(resultDir, exist_ok=True)

    # transform
    trainTransform = transforms.Compose([
        transforms.ToTensor()
    ])
    testTransform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    if args.inference != '':
        if not args.resume:
            print("args.resume must be set to do inference")
            exit(1)
        kwargs = {'num_workers': 1} if args.cuda else {}
        # src = args.inference
        # dst = args.save
        # inference_batch_size = args.ngpu
        # root = os.path.dirname(src)
        # images = os.path.basename(src)
        # dataset = dset.LUNA16(root=root, images=images, transform=testTransform, split=target_split, mode="infer")
        # loader = DataLoader(dataset, batch_size=inference_batch_size, shuffle=False, collate_fn=noop, **kwargs)
        # inference(args, loader, model, trainTransform)
        return

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}


    if args.data_format == "mhd":
        print("\nloading training set")
        dataManagerTrain = DM.DataManager(params['ModelParams']['dirTrain'],
                                            params['ModelParams']['dirResult'],
                                            params['DataManagerParams'])
        dataManagerTrain.loadTrainingData() # required
        numpyImages = dataManagerTrain.getNumpyImages()
        numpyGT = dataManagerTrain.getNumpyGT()

        trainSet = promise12.PROMISE12(mode='train', images=numpyImages, GT=numpyGT, transform=trainTransform, data_format=args.data_format)
        trainLoader = DataLoader(trainSet, batch_size=batch_size, shuffle=True, **kwargs)

        print("\nloading test set")
        dataManagerTest = DM.DataManager(params['ModelParams']['dirTest'],
                                        params['ModelParams']['dirResult'],
                                        params['DataManagerParams'])
        dataManagerTest.loadTestingData()  # required
        numpyImages = dataManagerTest.getNumpyImages()
        numpyGT = dataManagerTest.getNumpyGT()

        testSet = promise12.PROMISE12(mode='test', images=numpyImages, GT=numpyGT, transform=testTransform, data_format=args.data_format)
        testLoader = DataLoader(testSet, batch_size=batch_size, shuffle=True, **kwargs)
        
    elif args.data_format == "npy":
        
        if args.data_path:
            
            imagesTr_path = os.path.join(args.data_path, "imagesTr")
            numpyImages = load_npy_files_from_directory(imagesTr_path)
            
            labelsTr_path = os.path.join(args.data_path, "labelsTr")
            numpyGT = load_npy_files_from_directory(labelsTr_path)

            trainSet = promise12.PROMISE12(mode='train', images=numpyImages, GT=numpyGT, transform=trainTransform, data_format=args.data_format)
            trainLoader = DataLoader(trainSet, batch_size=batch_size, shuffle=True, **kwargs)


            imagesVal_path = os.path.join(args.data_path, "imagesVal")
            numpyImages = load_npy_files_from_directory(imagesVal_path)
            
            labelsVal_path = os.path.join(args.data_path, "labelsVal")
            numpyGT = load_npy_files_from_directory(labelsVal_path)
            
            valSet = promise12.PROMISE12(mode='test', images=numpyImages, GT=numpyGT, transform=testTransform, data_format=args.data_format)
            testLoader = DataLoader(valSet, batch_size=batch_size, shuffle=True, **kwargs)
        

    if args.opt == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=1e-1,
                              momentum=0.99, weight_decay=weight_decay)
    elif args.opt == 'adam':
        optimizer = optim.Adam(model.parameters(), weight_decay=weight_decay)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), weight_decay=weight_decay)

    trainF = open(os.path.join(resultDir, 'train.csv'), 'w')
    testF = open(os.path.join(resultDir, 'test.csv'), 'w')

    for epoch in range(1, args.nEpochs + 1):
        adjust_opt(args.opt, optimizer, epoch)
        
        train(args, epoch, model, trainLoader, optimizer, trainF)
        testDice = test(args, epoch, model, testLoader, optimizer, testF) # err is accuracy??? by Chao.
        
        if testDice < best_prec1:
            best_prec1 = testDice
        
            save_checkpoint({'epoch': epoch,
                            'state_dict': model.state_dict(),
                            'best_prec1': best_prec1}, 
                            path=resultDir, 
                            prefix="vnet"
                            )
            
    # os.system('./plot.py {} {} &'.format(len(trainLoader), resultDir))

    trainF.close()
    testF.close()

    # inference, i.e. output predicted mask for test data in .mhd
    # if args.inference != '':
        # if not args.resume:
        #     print("args.resume must be set to do inference")
        #     exit(1)
            
    if params['ModelParams']['dirInfer'] != '':
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(resultDir + "/vnet_checkpoint.pth.tar")
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
            print("loading inference data")
            
            if args.data_format == "mhd":
                dataManagerInfer = DM.DataManager(params['ModelParams']['dirInfer'],
                                                params['ModelParams']['dirResult'],
                                                params['DataManagerParams'])
                dataManagerInfer.loadInferData()  # required.  Create .loadInferData??? by Chao.
                numpyImages = dataManagerInfer.getNumpyImages()

                inferSet = promise12.PROMISE12(mode='infer', images=numpyImages, GT=None, transform=testTransform, data_format=args.data_format)
                inferLoader = DataLoader(inferSet, batch_size=batch_size, shuffle=True, **kwargs)
            
            elif args.data_format == "npy":
                
                imagesVal_path = os.path.join(args.data_path, "imagesVal")
                numpyImages = load_npy_files_from_directory(imagesVal_path)
                
                valSet = promise12.PROMISE12(mode='infer', images=numpyImages, GT=None, transform=testTransform, data_format=args.data_format)
                inferLoader = DataLoader(valSet, batch_size=batch_size, shuffle=True, **kwargs)

                
            inference(params, args, inferLoader, model, resultDir)


def train_dice(args, epoch, model, trainLoader, optimizer, trainF):
    model.train()
    nProcessed = 0
    nTrain = len(trainLoader.dataset)
    loss_list = []
    
    print("\nTrain loss --> Epoch " + str(epoch) + "\n")
    
    for batch_idx, output in enumerate(trainLoader):       
        
        data, target, id = output

        if args.loss == "dice":
            target = target.view(target.size(0),-1)       
            
        if args.cuda:
            data, target = data.cuda(), target.cuda()
            
        data = Variable(data)
        target = Variable(target)
        
        optimizer.zero_grad()
                
        output = model(data)
        
        if args.loss == "dice":

            output = output.permute(0, 2, 3, 4, 1).contiguous()
            output.view(output.size(0), output.numel() // (2 * output.size(0)), 2)
            
            loss_global = 0.0
            for i in range(output.size(0)):          
                        
                loss_loc = bioloss.dice_loss(output[i,:,:], target[i,:])
                loss_global += loss_loc.data[0].item()
            
            loss_global = loss_global / output.size(0)           
            loss = torch.tensor(loss_global, device='cuda' if args.cuda else 'cpu', requires_grad=True)        
            loss.data = torch.Tensor([loss_global])    
            loss_list.append(loss.data[0])
            
            print("\n\tFor trainning: Epoch: {} \t Loss per batches: {:.4f}", epoch, loss.item())
            
            loss_per_epoch = sum([loss.item() for loss in loss_list]) / len(loss_list)

        
        elif args.loss == "CE":
                
            print("\n\tCE loss for " + str(output.shape) + " prediction.")
              
            target = target.to(dtype=torch.long)
            """
            Parameters:
                weight (Tensor, optional) – a manual rescaling weight given to each class. If given, has to be a Tensor of size C and floating point dtype
                size_average (bool, optional) – Deprecated (see reduction). By default, the losses are averaged over each loss element in the batch. Note that for some losses, there are multiple elements per sample. If the field size_average is set to False, the losses are instead summed for each minibatch. Ignored when reduce is False. Default: True
                ignore_index (int, optional) – Specifies a target value that is ignored and does not contribute to the input gradient. When size_average is True, the loss is averaged over non-ignored targets. Note that ignore_index is only applicable when the target contains class indices.
                reduction (str, optional) – Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'. 'none': no reduction will be applied, 'mean': the weighted mean of the output is taken, 'sum': the output will be summed. Note: size_average and reduce are in the process of being deprecated, and in the meantime, specifying either of those two args will override reduction. Default: 'mean'
                label_smoothing (float, optional) – A float in [0.0, 1.0]. Specifies the amount of smoothing when computing the loss, where 0.0 means no smoothing. The targets become a mixture of the original ground truth and a uniform distribution as described in Rethinking the Inception Architecture for Computer Vision. Default: 0.0
            """
            loss_function = nn.CrossEntropyLoss(reduction='mean')
            loss = loss_function(output, target)
            
            # loss_loc = ce.cross_entropy_loss(output[i,:,:], target[i,:])
            loss_list.append(loss.item())
            
            loss_per_epoch = sum(loss_list) / len(loss_list)

            
            print("\tTrainning --> Epoch: {} \t Loss per batches: {:.4f}".format(epoch, loss.item()))
          
        elif args.loss == "diceCE":
            print("TODO fusion")
               
        loss.backward()
        optimizer.step()
        nProcessed += len(data)

    err_per_epoch = 100.*(1. - loss_per_epoch)
    
    print('\nFor trainning: Epoch: {} \tTotal loss: {:.4f}\tError: {:.4f}\n'.format(
    epoch, loss_per_epoch, err_per_epoch))
    
    if err_per_epoch == 100.0000:
        print("El loss ha caido mcho...")

        # trainF.write('{},{},{}\n'.format(partialEpoch, loss.data[0], err))
    trainF.write('{},{},{}\n'.format(epoch, loss_per_epoch, err_per_epoch))
    trainF.flush()

def test_dice(args, epoch, model, testLoader, optimizer, testF):
    model.eval()
    test_dice = 0
    incorrect = 0
    
    for batch_idx, output in enumerate(testLoader):
        data, target, id = output
        # print("testing with {}".format(id[0]))
        if args.cuda:
            data, target = data.cuda(), target.cuda()
            
        data = Variable(data)
        target = Variable(target)
        
        output = model(data)
        
        if args.loss == "dice":
            target = target.view(target.size(0),-1)       

            output = output.permute(0, 2, 3, 4, 1).contiguous()
            output.view(output.size(0), output.numel() // (2 * output.size(0)), 2)
            
            loss_global = 0.0
            for i in range(output.size(0)):          
                        
                loss_loc = bioloss.dice_loss(output[i,:,:], target[i,:])
                loss_global += loss_loc.data[0].item()
            
            loss_global = loss_global / output.size(0)           
            loss = torch.tensor(loss_global, device='cuda' if args.cuda else 'cpu', requires_grad=True)        
            loss.data = torch.Tensor([loss_global])    
            
            print("\n\tFor trainning: Epoch: {} \t Loss per batches: {:.4f}", epoch, loss.item())
            
            loss_global = loss_global / output.size(0)
        
            test_dice += loss_global
            incorrect += (1. - loss_global)
            
        elif args.loss == "CE":
                
            print("\n\tCE loss for " + str(output.shape) + " prediction.")
              
            target = target.to(dtype=torch.long)
            """
            Parameters:
                weight (Tensor, optional) – a manual rescaling weight given to each class. If given, has to be a Tensor of size C and floating point dtype
                size_average (bool, optional) – Deprecated (see reduction). By default, the losses are averaged over each loss element in the batch. Note that for some losses, there are multiple elements per sample. If the field size_average is set to False, the losses are instead summed for each minibatch. Ignored when reduce is False. Default: True
                ignore_index (int, optional) – Specifies a target value that is ignored and does not contribute to the input gradient. When size_average is True, the loss is averaged over non-ignored targets. Note that ignore_index is only applicable when the target contains class indices.
                reduction (str, optional) – Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'. 'none': no reduction will be applied, 'mean': the weighted mean of the output is taken, 'sum': the output will be summed. Note: size_average and reduce are in the process of being deprecated, and in the meantime, specifying either of those two args will override reduction. Default: 'mean'
                label_smoothing (float, optional) – A float in [0.0, 1.0]. Specifies the amount of smoothing when computing the loss, where 0.0 means no smoothing. The targets become a mixture of the original ground truth and a uniform distribution as described in Rethinking the Inception Architecture for Computer Vision. Default: 0.0
            """
            loss_function = nn.CrossEntropyLoss(reduction='mean')
            loss = loss_function(output, target)
            
            # loss_loc = ce.cross_entropy_loss(output[i,:,:], target[i,:])            
                    
            test_dice += loss.item()
            incorrect += (1. - loss.item())
            
            print("\tTrainning --> Epoch: {} \t Loss per batches: {:.4f}".format(epoch, loss.item()))
          
        elif args.loss == "diceCE":
            print("TODO fusion")

    nTotal = len(testLoader)
    test_dice /= nTotal  
    err = 100.*incorrect/nTotal

    print('\nFor testing: Epoch:{}\tAverage Dice Coeff: {:.4f}\tError:{:.4f}\n'.format(epoch, test_dice, err))

    testF.write('{},{},{}\n'.format(epoch, test_dice, err))
    testF.flush()
    return err

def adjust_opt(optAlg, optimizer, epoch):
    if optAlg == 'sgd':
        if epoch < 150:
            lr = 1e-1
        elif epoch == 150:
            lr = 1e-2
        elif epoch == 225:
            lr = 1e-3
        else:
            return

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr



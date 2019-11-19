# -*- coding: utf-8 -*-
import os
import sys
import json
import torch
import random
import logging
import argparse
import numpy as np
from tqdm import tqdm
from torch.nn import init
from config import config
from net import TrackerSiamRPN
from data import TrainDataLoader
from data_ir import TrainDataLoader_ir
from data_rgbt import TrainDataLoaderRGBT
from torch.utils.data import DataLoader
from util import util, AverageMeter, SavePlot
from got10k.datasets import ImageNetVID, GOT10k
from torchvision import datasets, transforms, utils
from custom_transforms import Normalize, ToTensor, RandomStretch, \
    RandomCrop, CenterCrop, RandomBlur, ColorAug
from experimentrgbt import RGBTSequence

torch.manual_seed(1234) # config.seed


parser = argparse.ArgumentParser(description='PyTorch SiameseRPN Training')

parser.add_argument('--train_path', default='/home/krautsct/RGB-T234', metavar='DIR',help='path to dataset')
parser.add_argument('--experiment_name', default='late_fusion', metavar='DIR',help='path to weight')
parser.add_argument('--checkpoint_path', default='./experiments/late_fusion/model/model_e50.pth', help='resume')
# /home/arbi/desktop/GOT-10k # /Users/arbi/Desktop # /home/arbi/desktop/ILSVRC
# 'experiments/default/model/model_e1.pth'
def main():

    '''parameter initialization'''
    args = parser.parse_args()
    exp_name_dir = util.experiment_name_dir(args.experiment_name)

    '''model on gpu'''
    model = TrackerSiamRPN()

    '''setup train data loader'''
    name = 'GOT-10k'
    assert name in ['VID', 'GOT-10k', 'All', 'RGBT-234']
    if name == 'GOT-10k':
        root_dir = args.train_path
        seq_dataset_rgb = GOT10k(root_dir, subset='train_i')
        seq_dataset_i = GOT10k(root_dir, subset='train_i', visible=False)
    elif name == 'VID':
        root_dir = '/home/arbi/desktop/ILSVRC'
        seq_dataset = ImageNetVID(root_dir, subset=('train'))
    elif name == 'All':
        root_dir_vid = '/home/arbi/desktop/ILSVRC'
        seq_datasetVID = ImageNetVID(root_dir_vid, subset=('train'))
        root_dir_got = args.train_path
        seq_datasetGOT = GOT10k(root_dir_got, subset='train')
        seq_dataset = util.data_split(seq_datasetVID, seq_datasetGOT)
    elif name == 'RGBT-234':
        root_dir = args.train_path
        seq_dataset = RGBTSequence(root_dir, subset='train')
        seq_dataset_val = RGBTSequence(root_dir, subset='val')
    print('seq_dataset', len(seq_dataset_rgb))

    train_z_transforms = transforms.Compose([
        ToTensor()
    ])
    train_x_transforms = transforms.Compose([
        ToTensor()
    ])

    train_data_ir  = TrainDataLoader_ir(seq_dataset_i, train_z_transforms, train_x_transforms, name)
    anchors = train_data_ir.anchors
    train_loader_ir = DataLoader(  dataset    = train_data_ir,
                                batch_size = config.train_batch_size,
                                shuffle    = True,
                                num_workers= config.train_num_workers,
                                pin_memory = True)
    train_data_rgb  = TrainDataLoader(seq_dataset_rgb, train_z_transforms, train_x_transforms, name)
    anchors = train_data_rgb.anchors
    train_loader_rgb = DataLoader(  dataset    = train_data_rgb,
                                batch_size = config.train_batch_size,
                                shuffle    = True,
                                num_workers= config.train_num_workers,
                                pin_memory = True)

    '''setup val data loader'''
    name = 'GOT-10k'
    assert name in ['VID', 'GOT-10k', 'All', 'RGBT-234']
    if name == 'GOT-10k':
        val_dir = '/home/krautsct/RGB-t-Val'
        seq_dataset_val_rgb = GOT10k(val_dir, subset='train_i')
        seq_dataset_val_ir = GOT10k(val_dir, subset='train_i', visible=False)
    elif name == 'VID':
        root_dir = '/home/arbi/desktop/ILSVRC'
        seq_dataset_val = ImageNetVID(root_dir, subset=('val'))
    elif name == 'All':
        root_dir_vid = '/home/arbi/desktop/ILSVRC'
        seq_datasetVID = ImageNetVID(root_dir_vid, subset=('val'))
        root_dir_got = args.train_path
        seq_datasetGOT = GOT10k(root_dir_got, subset='val')
        seq_dataset_val = util.data_split(seq_datasetVID, seq_datasetGOT)
    print('seq_dataset_val', len(seq_dataset_val_rgb))

    valid_z_transforms = transforms.Compose([
        ToTensor()
    ])
    valid_x_transforms = transforms.Compose([
        ToTensor()
    ])

    val_data  = TrainDataLoaderRGBT(seq_dataset_val_rgb, seq_dataset_val_ir, valid_z_transforms, valid_x_transforms, name)
    val_loader_ir = DataLoader(    dataset    = val_data,
                                batch_size = config.valid_batch_size,
                                shuffle    = False,
                                num_workers= config.valid_num_workers,
                                pin_memory = True)
    val_data_rgb  = TrainDataLoader(seq_dataset_val_rgb, valid_z_transforms, valid_x_transforms, name)
    val_loader_rgb = DataLoader(    dataset    = val_data_rgb,
                                batch_size = config.valid_batch_size,
                                shuffle    = False,
                                num_workers= config.valid_num_workers,
                                pin_memory = True)

    '''load weights'''

    if not args.checkpoint_path == None:
        assert os.path.isfile(args.checkpoint_path), '{} is not valid checkpoint_path'.format(args.checkpoint_path)
        checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
        if 'model' in checkpoint.keys():
            model.net.load_state_dict(torch.load(args.checkpoint_path, map_location='cpu')['model'])
        else:
            model.net.load_state_dict(torch.load(args.checkpoint_path, map_location='cpu'))
        torch.cuda.empty_cache()
        print('You are loading the model.load_state_dict')

    elif config.pretrained_model:
        checkpoint = torch.load(config.pretrained_model)
        # change name and load parameters
        checkpoint = {k.replace('features.features', 'featureExtract'): v for k, v in checkpoint.items()}
        model_dict = model.net.state_dict()
        model_dict.update(checkpoint)
        model.net.load_state_dict(model_dict)
        #torch.cuda.empty_cache()

    '''train phase'''
    train_closses, train_rlosses, train_tlosses = AverageMeter(), AverageMeter(), AverageMeter()
    val_closses, val_rlosses, val_tlosses = AverageMeter(), AverageMeter(), AverageMeter()

    train_val_plot = SavePlot(exp_name_dir, 'train_val_plot')

    for epoch in range(config.epoches):
        model.net.train()
        if config.fix_former_3_layers:
                util.freeze_layers(model.net)
        print('Train epoch {}/{}'.format(epoch+1, config.epoches))
        train_loss = []
        with tqdm(total=config.train_epoch_size) as progbar:
            for i, (dataset_rgb, dataset_ir) in enumerate(zip(train_loader_rgb, train_loader_ir)):

                closs, rloss, loss = model.step(epoch, dataset_rgb, dataset_ir, anchors, epoch, i, train=True)

                closs_ = closs.cpu().item()

                if np.isnan(closs_):
                   sys.exit(0)

                train_closses.update(closs.cpu().item())
                train_rlosses.update(rloss.cpu().item())
                train_tlosses.update(loss.cpu().item())

                progbar.set_postfix(closs='{:05.3f}'.format(train_closses.avg),
                                    rloss='{:05.5f}'.format(train_rlosses.avg),
                                    tloss='{:05.3f}'.format(train_tlosses.avg))

                progbar.update()
                train_loss.append(train_tlosses.avg)

                if i >= config.train_epoch_size - 1:

                    '''save model'''
                    model.save(model, exp_name_dir, epoch)

                    break

        train_loss = np.mean(train_loss)

        '''val phase'''
        val_loss = []
        with tqdm(total=config.val_epoch_size) as progbar:
            print('Val epoch {}/{}'.format(epoch+1, config.epoches))
            for i, (dataset_rgb, dataset_ir) in enumerate(zip(val_loader_rgb, val_loader_ir)):

                val_closs, val_rloss, val_tloss = model.step(epoch, dataset_rgb, dataset_ir, anchors, epoch, train=False)

                closs_ = val_closs.cpu().item()

                if np.isnan(closs_):
                    sys.exit(0)

                val_closses.update(val_closs.cpu().item())
                val_rlosses.update(val_rloss.cpu().item())
                val_tlosses.update(val_tloss.cpu().item())

                progbar.set_postfix(closs='{:05.3f}'.format(val_closses.avg),
                                    rloss='{:05.5f}'.format(val_rlosses.avg),
                                    tloss='{:05.3f}'.format(val_tlosses.avg))

                progbar.update()

                val_loss.append(val_tlosses.avg)

                if i >= config.val_epoch_size - 1:
                    break

        val_loss = np.mean(val_loss)
        train_val_plot.update(train_loss, val_loss)
        print ('Train loss: {}, val loss: {}'.format(train_loss, val_loss))


if __name__ == '__main__':
    main()

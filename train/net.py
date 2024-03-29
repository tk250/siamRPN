# -*- coding: utf-8 -*-

import os
import cv2
import torch
import random
import numpy as np
import torch.nn as nn
from util import util
import matplotlib.pyplot as plt
import torch.nn.functional as F
from config import config
from got10k.trackers import Tracker
from network_dense import SiameseAlexNet, SiameseAlexNetRGBT
from loss import rpn_smoothL1, rpn_cross_entropy_balance

class TrackerSiamRPN(Tracker):

    def __init__(self, net_path=None, **kargs):
        super(TrackerSiamRPN, self).__init__(
            name='SiamRPN', is_deterministic=True)

        '''setup GPU device if available'''
        self.cuda   = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.cuda else 'cpu')

        '''setup model'''
        self.net = SiameseAlexNet()
        #self.net.init_weights()

        if net_path is not None:
            self.net.load_state_dict(torch.load(
                net_path, map_location = lambda storage, loc: storage ))
        if self.cuda:
            self.net = self.net.to(self.device)

        '''setup optimizer'''
        self.optimizer   = torch.optim.SGD(
            self.net.parameters(),
            lr           = config.lr,
            momentum     = config.momentum,
            weight_decay = config.weight_decay)

    def step(self, epoch, dataset_rgb, anchors, epoche, i = 0,  train=True):

        if train:
            self.net.train()
        else:
            self.net.eval()

        #template_i, detection_i, _, _ = dataset_ir
        template_rgb, detection_rgb, regression_target, conf_target = dataset_rgb
        #print('stuff', template_rgb.mean(), detection_rgb.mean(), regression_target.mean(), conf_target.mean())
        #template_i, detection_i = torch.from_numpy(np.zeros(template_i.size())).float(), torch.from_numpy(np.zeros(detection_i.size())).float()

        if self.cuda:
            template_rgb, detection_rgb = template_rgb.cuda(), detection_rgb.cuda()#, template_i, detection_i, template_i.cuda(), detection_i.cuda()
            regression_target, conf_target = regression_target.cuda(), conf_target.cuda()
        pred_score, pred_regression = self.net(template_rgb, detection_rgb)#, template_i, detection_i
        #print('stuff', pred_score.mean(), pred_regression.mean())

        pred_conf   = pred_score.reshape(-1, 2, config.size).permute(0, 2, 1)

        pred_offset = pred_regression.reshape(-1, 4, config.size).permute(0, 2, 1)

        cls_loss = rpn_cross_entropy_balance(   pred_conf,
                                                conf_target,
                                                config.num_pos,
                                                config.num_neg,
                                                anchors,
                                                ohem_pos=config.ohem_pos,
                                                ohem_neg=config.ohem_neg)

        reg_loss = rpn_smoothL1(pred_offset,
                                regression_target,
                                conf_target,
                                config.num_pos,
                                ohem=config.ohem_reg)

        loss = cls_loss + config.lamb * reg_loss

        '''anchors_show = anchors
        exem_img = template[0].cpu().numpy().transpose(1, 2, 0)  # (127, 127, 3)
        #cv2.imwrite('exem_img.png', exem_img)

        inst_img = detection[0].cpu().numpy().transpose(1, 2, 0) # (255, 255, 3)
        #cv2.imwrite('inst_img.png', inst_img)



        topk = 1
        cls_pred = F.softmax(pred_conf, dim=2)[0, :, 1]

        topk_box = util.get_topk_box(cls_pred, pred_offset[0], anchors_show, topk=topk)
        img_box = util.add_box_img(inst_img, topk_box, color=(0, 0, 255))

        cv2.imwrite('pred_inst.png', img_box)

        cls_pred = conf_target[0]
        gt_box = util.get_topk_box(cls_pred, regression_target[0], anchors_show)
        #print('gt_box', gt_box)
        img_box = util.add_box_img(img_box, gt_box, color=(255, 0, 0), x = 1, y = 1)
        #print('gt_box', gt_box)
        cv2.imwrite('pred_inst_gt.png', img_box)'''

        if train:
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), config.clip)
            self.optimizer.step()

        return cls_loss, reg_loss, loss

    '''save model'''
    def save(self,model, exp_name_dir, epoch):
        util.adjust_learning_rate(self.optimizer, config.gamma)

        model_save_dir_pth = '{}/model'.format(exp_name_dir)
        if not os.path.exists(model_save_dir_pth):
                os.makedirs(model_save_dir_pth)
        net_path = os.path.join(model_save_dir_pth, 'model_e%d.pth' % (epoch + 1))
        torch.save(model.net.state_dict(), net_path)

'''class SiamRPN(nn.Module):

    def __init__(self, anchor_num = 5):
        super(SiamRPN, self).__init__()

        self.anchor_num = anchor_num
        self.feature = nn.Sequential(
            # conv1
            nn.Conv2d(3, 64, kernel_size = 11, stride = 2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 3, stride = 2),
            # conv2
            nn.Conv2d(64, 192, kernel_size = 5),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size = 3, stride = 2),
            # conv3
            nn.Conv2d(192, 384, kernel_size = 3),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace = True),
            # conv4
            nn.Conv2d(384, 256, kernel_size = 3),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace = True),
            # conv5
            nn.Conv2d(256, 256, kernel_size = 3),
            nn.BatchNorm2d(256))

        self.conv_reg_z = nn.Conv2d(256, 256 * 4 * self.anchor_num, 3, 1)
        self.conv_reg_x = nn.Conv2d(256, 256, 3)
        self.conv_cls_z = nn.Conv2d(256, 256 * 2 * anchor_num, 3, 1)
        self.conv_cls_x = nn.Conv2d(256, 256, 3)
        self.adjust_reg = nn.Conv2d(4 * anchor_num, 4 * anchor_num*1, 1)

    def forward(self, z, x):
        return self.inference(x, *self.learn(z))

    def learn(self, z):
        z = self.feature(z)
        kernel_reg = self.conv_reg_z(z)
        kernel_cls = self.conv_cls_z(z)

        k = kernel_reg.size()[-1]
        kernel_reg = kernel_reg.view(4 * self.anchor_num, 256, k, k)
        kernel_cls = kernel_cls.view(2 * self.anchor_num, 256, k, k)

        return kernel_reg, kernel_cls

    def inference(self, x, kernel_reg, kernel_cls):
        x = self.feature(x)
        x_reg = self.conv_reg_x(x)
        x_cls = self.conv_cls_x(x)

        out_reg = self.adjust_reg(F.conv2d(x_reg, kernel_reg))
        out_cls = F.conv2d(x_cls, kernel_cls)

        return out_reg, out_cls'''

import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms

from torchvision.models import alexnet
from torch.autograd import Variable
from torch import nn

from config import config

class SiameseAlexNet(nn.Module):
    def __init__(self, ):
        super(SiameseAlexNet, self).__init__()
        self.featureExtract = nn.Sequential(
            nn.Conv2d(3, 96, 11, stride=2),
            nn.BatchNorm2d(96),
            nn.MaxPool2d(3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 256, 5),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 384, 3),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, 3),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, 3),
            nn.BatchNorm2d(256),
        )
        self.anchor_num = config.anchor_num
        self.input_size = config.detection_img_size
        self.score_displacement = int((self.input_size - config.template_img_size) / config.total_stride)
        self.conv_cls1 = nn.Conv2d(256, 256 * 2 * self.anchor_num, kernel_size=3, stride=1, padding=0)
        self.conv_r1 = nn.Conv2d(256, 256 * 4 * self.anchor_num, kernel_size=3, stride=1, padding=0)

        self.conv_cls2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0)
        self.conv_r2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0)
        self.regress_adjust = nn.Conv2d(4 * self.anchor_num, 4 * self.anchor_num, 1)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, std= 0.0005)
                nn.init.normal_(m.bias.data, std= 0.0005)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, template, detection):
        N = template.size(0)
        template_feature = self.featureExtract(template)
        detection_feature = self.featureExtract(detection)

        kernel_score = self.conv_cls1(template_feature).view(N, 2 * self.anchor_num, 256, 4, 4)
        kernel_regression = self.conv_r1(template_feature).view(N, 4 * self.anchor_num, 256, 4, 4)
        conv_score = self.conv_cls2(detection_feature)
        conv_regression = self.conv_r2(detection_feature)

        conv_scores = conv_score.reshape(1, -1, self.score_displacement + 4, self.score_displacement + 4)
        score_filters = kernel_score.reshape(-1, 256, 4, 4)
        pred_score = F.conv2d(conv_scores, score_filters, groups=N).reshape(N, 10, self.score_displacement + 1,
                                                                            self.score_displacement + 1)

        conv_reg = conv_regression.reshape(1, -1, self.score_displacement + 4, self.score_displacement + 4)
        reg_filters = kernel_regression.reshape(-1, 256, 4, 4)
        pred_regression = self.regress_adjust(
            F.conv2d(conv_reg, reg_filters, groups=N).reshape(N, 20, self.score_displacement + 1,
                                                              self.score_displacement + 1))
        return pred_score, pred_regression

    def track_init(self, template):
        print('mean', template.mean())
        N = template.size(0)
        template_feature = self.featureExtract(template)

        kernel_score = self.conv_cls1(template_feature).view(N, 2 * self.anchor_num, 256, 4, 4)
        kernel_regression = self.conv_r1(template_feature).view(N, 4 * self.anchor_num, 256, 4, 4)
        self.score_filters = kernel_score.reshape(-1, 256, 4, 4)
        self.reg_filters = kernel_regression.reshape(-1, 256, 4, 4)

    def track(self, detection):
        N = detection.size(0)
        detection_feature = self.featureExtract(detection)

        conv_score = self.conv_cls2(detection_feature)
        conv_regression = self.conv_r2(detection_feature)

        conv_scores = conv_score.reshape(1, -1, self.score_displacement + 4, self.score_displacement + 4)
        pred_score = F.conv2d(conv_scores, self.score_filters, groups=N).reshape(N, 10, self.score_displacement + 1,
                                                                                 self.score_displacement + 1)
        conv_reg = conv_regression.reshape(1, -1, self.score_displacement + 4, self.score_displacement + 4)
        pred_regression = self.regress_adjust(
            F.conv2d(conv_reg, self.reg_filters, groups=N).reshape(N, 20, self.score_displacement + 1,
                                                                   self.score_displacement + 1))
        return pred_score, pred_regression

class SiameseAlexNetRGBT(nn.Module):
    def __init__(self, ):
        super(SiameseAlexNetRGBT, self).__init__()
        #RGB BRANCH
        self.conv1_rgb = nn.Sequential(
            nn.Conv2d(3, 96, 11, 2),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.MaxPool2d(3, 2))
        self.conv2_rgb = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1, groups=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.MaxPool2d(3, 2))
        self.conv3_rgb = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1))
        self.conv4_rgb = nn.Sequential(
            nn.Conv2d(384, 384, 3, 1, groups=2),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1))

        #IR BRANCH
        self.conv1_ir = nn.Sequential(
            nn.Conv2d(1, 96, 11, 2),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))
        self.conv2_ir = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1, groups=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))
        self.conv3_ir = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True))
        self.conv4_ir = nn.Sequential(
            nn.Conv2d(384, 384, 3, 1, groups=2),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True))


        # Shared layers
        self.conv1_shared = nn.Sequential(
            nn.Conv2d(4, 256, 38, 8),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))
        self.conv3_shared = nn.Sequential(
            nn.Conv2d(768, 384, 5, 1, groups=2),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(
            nn.Conv2d(1152, 256, 3, 1, groups=2),
            nn.BatchNorm2d(256))


        self.anchor_num = config.anchor_num
        self.input_size = config.detection_img_size
        self.score_displacement = int((self.input_size - config.template_img_size) / config.total_stride)
        self.conv_cls1 = nn.Conv2d(256, 256 * 2 * self.anchor_num, kernel_size=3, stride=1, padding=0)
        self.conv_r1 = nn.Conv2d(256, 256 * 4 * self.anchor_num, kernel_size=3, stride=1, padding=0)

        self.conv_cls2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0)
        self.conv_r2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0)
        self.regress_adjust = nn.Conv2d(4 * self.anchor_num, 4 * self.anchor_num, 1)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, std= 0.0005)
                nn.init.normal_(m.bias.data, std= 0.0005)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, template_rgb, detection_rgb, template_ir, detection_ir):
        N = template_rgb.size(0)
        x_shared1_temp = torch.cat((template_rgb, template_ir), 1)
        x_shared1_temp = self.conv1_shared(x_shared1_temp)

        # rgb branch
        x_rgb_temp = self.conv1_rgb(template_rgb)
        x_rgb_temp = self.conv2_rgb(x_rgb_temp)

        # ir branch
        x_v_temp = self.conv1_ir(template_ir)
        x_v_temp = self.conv2_ir(x_v_temp)

        x_shared_temp = torch.cat((x_rgb_temp, x_shared1_temp, x_v_temp), 1)
        x_shared_temp = self.conv3_shared(x_shared_temp)


        x_rgb_temp = self.conv3_rgb(x_rgb_temp)
        x_rgb_temp = self.conv4_rgb(x_rgb_temp)


        x_v_temp = self.conv3_ir(x_v_temp)
        x_v_temp = self.conv4_ir(x_v_temp)
        x_final_temp = torch.cat((x_rgb_temp, x_shared_temp, x_v_temp), 1)
        template_feature = self.conv5(x_final_temp)

        x_shared1_det = torch.cat((detection_rgb, detection_ir), 1)
        x_shared1_det = self.conv1_shared(x_shared1_det)

        # rgb branch
        x_rgb_det = self.conv1_rgb(detection_rgb)
        x_rgb_det = self.conv2_rgb(x_rgb_det)

        # ir branch
        x_v_det = self.conv1_ir(detection_ir)
        x_v_det = self.conv2_ir(x_v_det)

        x_shared_det = torch.cat((x_rgb_det, x_shared1_det, x_v_det), 1)
        x_shared_det = self.conv3_shared(x_shared_det)


        x_rgb_det = self.conv3_rgb(x_rgb_det)
        x_rgb_det = self.conv4_rgb(x_rgb_det)


        x_v_det = self.conv3_ir(x_v_det)
        x_v_det = self.conv4_ir(x_v_det)
        x_final_det = torch.cat((x_rgb_det, x_shared_det, x_v_det), 1)
        detection_feature = self.conv5(x_final_det)


        kernel_score = self.conv_cls1(template_feature).view(N, 2 * self.anchor_num, 256, 4, 4)
        kernel_regression = self.conv_r1(template_feature).view(N, 4 * self.anchor_num, 256, 4, 4)
        conv_score = self.conv_cls2(detection_feature)
        conv_regression = self.conv_r2(detection_feature)
        conv_scores = conv_score.reshape(1, -1, self.score_displacement + 4, self.score_displacement + 4)
        score_filters = kernel_score.reshape(-1, 256, 4, 4)
        pred_score = F.conv2d(conv_scores, score_filters, groups=N).reshape(N, 10, self.score_displacement + 1,
                                                                            self.score_displacement + 1)

        conv_reg = conv_regression.reshape(1, -1, self.score_displacement + 4, self.score_displacement + 4)
        reg_filters = kernel_regression.reshape(-1, 256, 4, 4)
        pred_regression = self.regress_adjust(
            F.conv2d(conv_reg, reg_filters, groups=N).reshape(N, 20, self.score_displacement + 1,
                                                              self.score_displacement + 1))
        return pred_score, pred_regression

    def track_init(self, template_rgb, template_ir):
        N = template_rgb.size(0)
        x_shared1_temp = torch.cat((template_rgb, template_ir), 1)
        x_shared1_temp = self.conv1_shared(x_shared1_temp)

        # rgb branch
        x_rgb_temp = self.conv1_rgb(template_rgb)
        x_rgb_temp = self.conv2_rgb(x_rgb_temp)

        # ir branch
        x_v_temp = self.conv1_ir(template_ir)
        x_v_temp = self.conv2_ir(x_v_temp)

        x_shared_temp = torch.cat((x_rgb_temp, x_shared1_temp, x_v_temp), 1)
        x_shared_temp = self.conv3_shared(x_shared_temp)


        x_rgb_temp = self.conv3_rgb(x_rgb_temp)
        x_rgb_temp = self.conv4_rgb(x_rgb_temp)


        x_v_temp = self.conv3_ir(x_v_temp)
        x_v_temp = self.conv4_ir(x_v_temp)
        x_final_temp = torch.cat((x_rgb_temp, x_shared_temp, x_v_temp), 1)
        template_feature = self.conv5(x_final_temp)

        kernel_score = self.conv_cls1(template_feature).view(N, 2 * self.anchor_num, 256, 4, 4)
        kernel_regression = self.conv_r1(template_feature).view(N, 4 * self.anchor_num, 256, 4, 4)
        self.score_filters = kernel_score.reshape(-1, 256, 4, 4)
        self.reg_filters = kernel_regression.reshape(-1, 256, 4, 4)

    def track(self, detection_rgb, detection_ir):
        N = detection_rgb.size(0)
        x_shared1_det = torch.cat((detection_rgb, detection_ir), 1)
        x_shared1_det = self.conv1_shared(x_shared1_det)

        # rgb branch
        x_rgb_det = self.conv1_rgb(detection_rgb)
        x_rgb_det = self.conv2_rgb(x_rgb_det)

        # ir branch
        x_v_det = self.conv1_ir(detection_ir)
        x_v_det = self.conv2_ir(x_v_det)

        x_shared_det = torch.cat((x_rgb_det, x_shared1_det, x_v_det), 1)
        x_shared_det = self.conv3_shared(x_shared_det)


        x_rgb_det = self.conv3_rgb(x_rgb_det)
        x_rgb_det = self.conv4_rgb(x_rgb_det)


        x_v_det = self.conv3_ir(x_v_det)
        x_v_det = self.conv4_ir(x_v_det)
        x_final_det = torch.cat((x_rgb_det, x_shared_det, x_v_det), 1)
        detection_feature = self.conv5(x_final_det)

        conv_score = self.conv_cls2(detection_feature)
        conv_regression = self.conv_r2(detection_feature)

        conv_scores = conv_score.reshape(1, -1, self.score_displacement + 4, self.score_displacement + 4)
        pred_score = F.conv2d(conv_scores, self.score_filters, groups=N).reshape(N, 10, self.score_displacement + 1,
                                                                                 self.score_displacement + 1)
        conv_reg = conv_regression.reshape(1, -1, self.score_displacement + 4, self.score_displacement + 4)
        pred_regression = self.regress_adjust(
            F.conv2d(conv_reg, self.reg_filters, groups=N).reshape(N, 20, self.score_displacement + 1,
                                                                   self.score_displacement + 1))
        return pred_score, pred_regression

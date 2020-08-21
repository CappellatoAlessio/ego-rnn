import torch
import resnetMod
import torch.nn as nn
from torch.nn import functional as F
# from torch.autograd import Variable
from MyConvLSTMCell import *


class attentionModel(nn.Module):
    def __init__(self, num_classes=61, mem_size=512):
        super(attentionModel, self).__init__()
        self.num_classes = num_classes
        self.resNet = resnetMod.resnet34(True, True)
        self.mem_size = mem_size
        self.weight_softmax = self.resNet.fc.weight
        self.lstm_cell = MyConvLSTMCell(512, mem_size)
        self.avgpool = nn.AvgPool2d(7)
        self.dropout = nn.Dropout(0.7)
        self.fc = nn.Linear(mem_size, self.num_classes)
        self.classifier = nn.Sequential(self.dropout, self.fc)

        self.ss_relu = nn.ReLU()
        self.ss_bn = nn.BatchNorm2d(100)
        self.ss_conv = nn.Conv2d(512, 100, 1)
        self.ss_fc = nn.Linear(100*7*7, 1*7*7)
        

    def forward(self, inputVariable):
        state = (torch.zeros((inputVariable.size(1), self.mem_size, 7, 7)).cuda(),
                 torch.zeros((inputVariable.size(1), self.mem_size, 7, 7)).cuda())

        ss_feats = []

        for t in range(inputVariable.size(0)):
            logit, feature_conv, feature_convNBN = self.resNet(inputVariable[t])
            bz, nc, h, w = feature_conv.size()
            feature_conv1 = feature_conv.view(bz, nc, h*w)
            probs, idxs = logit.sort(1, True)
            class_idx = idxs[:, 0]
            cam = torch.bmm(self.weight_softmax[class_idx].unsqueeze(1), feature_conv1)
            attentionMAP = F.softmax(cam.squeeze(1), dim=1)
            attentionMAP = attentionMAP.view(attentionMAP.size(0), 1, 7, 7)
            attentionFeat = feature_convNBN * attentionMAP.expand_as(feature_conv)
            state = self.lstm_cell(attentionFeat, state)


            ss_x = self.ss_conv(attentionFeat)
            ss_x = self.ss_relu(self.ss_bn(ss_x))
            ss_bz, ss_nc, ss_h, ss_w = ss_x.size()
            ss_x = ss_x.view(ss_bz, ss_nc*ss_h*ss_w)
            ss_x = self.ss_fc(ss_x)
            ss_x = ss_x.view(ss_bz, -1, ss_h, ss_w)
            ss_feats.append(ss_x)

        feats1 = self.avgpool(state[1]).view(state[1].size(0), -1)
        feats = self.classifier(feats1)

        ss_feats = torch.stack(ss_feats, 2)
        ss_feats = ss_feats.view(ss_bz, -1, ss_h, ss_w)

        return feats, feats1, ss_feats

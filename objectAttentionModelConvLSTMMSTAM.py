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

        self.GAP1 = nn.AdaptiveAvgPool2d((1,1))
        self.GAP2 = nn.AdaptiveAvgPool2d((1,1))

        self.fc_att1 = nn.Linear(mem_size*2, mem_size)
        self.fc_att2 = nn.Linear(mem_size, mem_size)


    def forward(self, inputVariable):
        state = (torch.zeros((inputVariable.size(1), self.mem_size, 7, 7)).cuda(),
                 torch.zeros((inputVariable.size(1), self.mem_size, 7, 7)).cuda())

        ss_feats = []

        for t in range(inputVariable.size(0)):
            logit, feature_conv, feature_convNBN = self.resNet(inputVariable[t])
            

            visualFeatStat = self.GAP1(feature_convNBN)
            hiddenStat = self.GAP2(state[0])

            attRaw = torch.cat([visualFeatStat, hiddenStat], dim=1)
            attRaw = attRaw.view(attRaw.shape[0], attRaw.shape[1])

            attWeights = torch.sigmoid(self.fc_att2(F.relu(self.fc_att1(attRaw))))
            attentionFeat = feature_conv*attWeights.view(attWeights.shape[0], attWeights.shape[1], 1, 1)

            state = self.lstm_cell(attentionFeat, state)

            ss_x = self.ss_conv(feature_convNBN)
            ss_x = self.ss_relu(self.ss_bn(ss_x))
            ss_bz, ss_nc, ss_h, ss_w = ss_x.size()
            ss_x = ss_x.view(ss_bz, ss_nc*ss_h*ss_w)
            ss_x = self.ss_fc(ss_x)
            ss_x = ss_x.view(ss_bz, -1, ss_h, ss_w)
            # ss_x = self.ss_softmax(ss_x)
            ss_feats.append(ss_x)

        feats1 = self.avgpool(state[1]).view(state[1].size(0), -1)
        feats = self.classifier(feats1)

        ss_feats = torch.stack(ss_feats, 2)
        ss_feats = ss_feats.view(ss_bz, -1, ss_h, ss_w)

        return feats, feats1, ss_feats
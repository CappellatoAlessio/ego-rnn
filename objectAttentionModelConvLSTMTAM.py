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

        self.GAP1 = nn.AdaptiveAvgPool2d((1,1))
        self.GAP2 = nn.AdaptiveAvgPool2d((1,1))

        self.fc_att1 = nn.Linear(mem_size*2, mem_size)
        self.fc_att2 = nn.Linear(mem_size, mem_size)

    def forward(self, inputVariable):
        state = (torch.zeros((inputVariable.size(1), self.mem_size, 7, 7)).cuda(),
                 torch.zeros((inputVariable.size(1), self.mem_size, 7, 7)).cuda())
        for t in range(inputVariable.size(0)):
            logit, feature_conv, feature_convNBN = self.resNet(inputVariable[t])

            visualFeatStat = self.GAP1(feature_convNBN)
            hiddenStat = self.GAP2(state[0])

            attRaw = torch.cat([visualFeatStat, hiddenStat], dim=1)
            attRaw = attRaw.view(attRaw.shape[0], attRaw.shape[1])

            attWeights = torch.sigmoid(self.fc_att2(F.relu(self.fc_att1(attRaw))))
            attentionFeat = feature_convNBN*attWeights.view(attWeights.shape[0], attWeights.shape[1], 1, 1)

            state = self.lstm_cell(attentionFeat, state)

        feats1 = self.avgpool(state[1]).view(state[1].size(0), -1)
        feats = self.classifier(feats1)
        return feats, feats1
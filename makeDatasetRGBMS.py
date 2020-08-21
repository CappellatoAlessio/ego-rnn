import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import glob
import random


def gen_split(root_dir, stackSize):
    Dataset = []
    mmaps = []
    Labels = []
    NumFrames = []
    root_dir = os.path.join(root_dir, 'frames')
    for dir_user in sorted(os.listdir(root_dir)):
        class_id = 0
        dir = os.path.join(root_dir, dir_user)
        for target in sorted(os.listdir(dir)):
            dir1 = os.path.join(dir, target)
            insts = sorted(os.listdir(dir1))
            if insts != []:
                for inst in insts:
                    inst_dir = os.path.join(dir1, inst)
                    dir_rgb = os.path.join(inst_dir, "rgb")
                    dir_mmaps = os.path.join(inst_dir, "mmaps")
                    numFrames = len(glob.glob1(dir_rgb, '*.png'))
                    if numFrames >= stackSize:
                        Dataset.append(dir_rgb)
                        mmaps.append(dir_mmaps)
                        Labels.append(class_id)
                        NumFrames.append(numFrames)
            class_id += 1
    return Dataset, mmaps, Labels, NumFrames

class makeDataset(Dataset):
    def __init__(self, root_dir, spatial_transform=None, mmaps_transform=None, seqLen=20,
                 train=True, mulSeg=False, numSeg=1, fmt='.png'):

        self.images, self.mmaps, self.labels, self.numFrames = gen_split(root_dir, 5)
        self.spatial_transform = spatial_transform
        self.mmaps_transform = mmaps_transform
        self.train = train
        self.mulSeg = mulSeg
        self.numSeg = numSeg
        self.seqLen = seqLen
        self.fmt = fmt

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        vid_name = self.images[idx]
        mmaps_name = self.mmaps[idx]
        label = self.labels[idx]
        numFrame = self.numFrames[idx]
        inpSeq = []
        mmapsSeq= []

        self.spatial_transform.randomize_parameters()
        if self.train:
          self.mmaps_transform.transforms[1].p = self.spatial_transform.transforms[1].p
          self.mmaps_transform.transforms[2].scale = self.spatial_transform.transforms[2].scale
          self.mmaps_transform.transforms[2].crop_position = self.spatial_transform.transforms[2].crop_position

        for i in np.linspace(1, numFrame, self.seqLen, endpoint=False):
            fl_name = vid_name + '/' + 'rgb' + str(int(np.floor(i))).zfill(4) + self.fmt
            fl_mmaps_name = mmaps_name + '/' + 'map' + str(int(np.floor(i))).zfill(4) + self.fmt
            img = Image.open(fl_name)
            mmaps = Image.open(fl_mmaps_name)
            
            inpSeq.append(self.spatial_transform(img.convert('RGB')))
            mmapsSeq.append(self.mmaps_transform(mmaps.convert('1')))
        inpSeq = torch.stack(inpSeq, 0)
        mmapsSeq = torch.stack(mmapsSeq, 0)
        return inpSeq, mmapsSeq, label

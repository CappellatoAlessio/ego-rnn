import argparse
import glob
import imageio
from pathlib import Path

import cv2
import numpy as np
import sys
from PIL import Image
from torchvision import transforms



def gen(sampleDir, seqLen, outDir):


    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    preprocess1 = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
    ])

    preprocess2 = transforms.Compose([
        transforms.ToTensor(),
        normalize])

    attentionMap_images = []
    for i in np.linspace(1, len(glob.glob1(sampleDir, '*.png')), seqLen, endpoint=False):
        fl_name_in = sampleDir + '/' + 'rgb' + str(int(np.floor(i))).zfill(4) + '.png'
        fl_name_out = outDir + '/' + 'rgb' + str(int(np.floor(i))).zfill(4) + '_attention' + '.png'
        img_pil = Image.open(fl_name_in)
        img_pil1 = preprocess1(img_pil)
        img_size = img_pil1.size
        img = np.asarray(img_pil1)
        img = cv2.cvtColor(np.uint8(img), cv2.COLOR_RGB2BGR)
        cv2.imwrite(fl_name_out, img)


def __main__():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sampleDir', type=str, default='./dataset/gtea_warped_flow_61/split2/train',
                        help='Sample directory')
    parser.add_argument('--seqLen', type=int, default=25, help='Length of sequence')
    parser.add_argument('--outDir', type=str, default='experiments', help='Directory to save results')

    args = parser.parse_args()

    sampleDir = args.sampleDir
    outDir = args.outDir
    seqLen = args.seqLen


    gen(sampleDir, seqLen, outDir)


__main__()

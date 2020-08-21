import argparse
import glob
import imageio
from pathlib import Path

import cv2
import numpy as np
import sys
from PIL import Image
from torchvision import transforms


from attentionMapModel import attentionMap
from objectAttentionModelConvLSTM import *


def gen(dataset, sampleDir, modelDict, outDir, seqLen, memSize, genGif):
    ####################Model definition###############################
    if dataset == 'gtea61':
        num_classes = 61
    elif dataset == 'gtea71':
        num_classes = 71
    elif dataset == 'gtea_gaze':
        num_classes = 44
    elif dataset == 'egtea':
        num_classes = 106
    else:
        print('Dataset not found')
        sys.exit()

    model = attentionModel(num_classes=num_classes, mem_size=memSize)
    model.load_state_dict(torch.load(modelDict), strict=False)
    model_backbone = model.resNet
    attentionMapModel = attentionMap(model_backbone).cuda()
    attentionMapModel.train(False)
    for params in attentionMapModel.parameters():
        params.requires_grad = False
    ###################################################################

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
        size_upsample = (img_size[0], img_size[1])
        img_tensor = preprocess2(img_pil1)
        img_variable = img_tensor.unsqueeze(0).cuda()
        img = np.asarray(img_pil1)
        attentionMap_image = attentionMapModel(img_variable, img, size_upsample)
        cv2.imwrite(fl_name_out, attentionMap_image)
    
    if genGif:
        fl_gif_out = outDir + '/' + 'gif' + '_attention' + '.gif'
        image_path = Path(outDir)
        images = list(image_path.glob('*.png'))
        image_list = []
        for file_name in images:
            image_list.append(imageio.imread(file_name))  
             
        imageio.mimwrite(fl_gif_out, image_list, format='GIF', duration=seqLen*0.05)


def __main__():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='gtea61', help='Dataset')
    parser.add_argument('--sampleDir', type=str, default='./dataset/gtea_warped_flow_61/split2/train',
                        help='Sample directory')
    parser.add_argument('--outDir', type=str, default='experiments', help='Directory to save results')
    parser.add_argument('--modelDict', type=str, default='./experiments/gtea61/rgb+ms/stage2/model_rgb_state_dict.pth',
                        help='Model path')
    parser.add_argument('--seqLen', type=int, default=25, help='Length of sequence')
    parser.add_argument('--memSize', type=int, default=512, help='ConvLSTM hidden state size')
    parser.add_argument('--genGif', type=bool, default=True, help='Set to true if want to generate also GIF')

    args = parser.parse_args()

    dataset = args.dataset
    sampleDir = args.sampleDir
    outDir = args.outDir
    modelDict = args.modelDict
    seqLen = args.seqLen
    memSize = args.memSize
    genGif = args.genGif

    gen(dataset, sampleDir, modelDict, outDir, seqLen, memSize, genGif)


__main__()

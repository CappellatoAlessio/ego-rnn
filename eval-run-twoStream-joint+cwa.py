from __future__ import print_function, division

import argparse

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from makeDatasetTwoStream import *
from spatial_transforms import (Compose, ToTensor, CenterCrop, Scale, Normalize)
# from torch.autograd import Variable
from twoStreamModelCWA import *


def getClassNames(dataset_dir):
    return np.array(sorted(os.listdir(os.path.join(dataset_dir, 'frames/S2'))))


def plotAccuracyPerClassHist(accuracies, dataset, dataset_dir):
    class_names = getClassNames(dataset_dir)
    df = pd.DataFrame(data={'classes': class_names, 'accs': accuracies})
    df = df.sort_values(by=["accs"], ascending=False)

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.barplot(y=df["classes"], x=df["accs"], ax=ax)
    plt.savefig(dataset + 'class_hist.png', bbox_inches='tight')
    plt.show()


def plotConfMatr(conf_matr, dataset, dataset_dir):
    class_names = getClassNames(dataset_dir)
    fig, ax = plt.subplots(figsize=(31, 31))
    ax = sns.heatmap(conf_matr, annot=True, fmt='.2g', linewidth=0.1, xticklabels=class_names, square=True)
    ax.set_yticklabels(labels=class_names, rotation=0)
    plt.savefig(dataset + 'conf_matr.png', bbox_inches='tight')
    plt.show()


def main_run(dataset, model_state_dict, dataset_dir, stackSize, seqLen, memSize, variant):
    if dataset == 'gtea61':
        num_classes = 61
    elif dataset == 'gtea71':
        num_classes = 71
    elif dataset == 'gtea_gaze':
        num_classes = 44
    elif dataset == 'egtea':
        num_classes = 106

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    normalize = Normalize(mean=mean, std=std)

    testBatchSize = 1
    spatial_transform = Compose([Scale(256), CenterCrop(224), ToTensor(), normalize])

    vid_seq_test = makeDataset(dataset_dir, spatial_transform=spatial_transform, sequence=False, numSeg=1,
                               stackSize=stackSize, fmt='.png', phase='Test', seqLen=seqLen)

    test_loader = torch.utils.data.DataLoader(vid_seq_test, batch_size=testBatchSize,
                                              shuffle=False, num_workers=2, pin_memory=True)

    model = twoStreamAttentionModel(variant, stackSize=5, memSize=512, num_classes=num_classes)
    model.load_state_dict(torch.load(model_state_dict))

    for params in model.parameters():
        params.requires_grad = False

    model.train(False)
    model.cuda()

    test_samples = vid_seq_test.__len__()
    print('Number of samples = {}'.format(test_samples))
    print('Evaluating...')
    numCorrTwoStream = 0

    predicted_labels = []
    true_labels = []
    all_predicted = []
    for j, (inputFlow, inputFrame, targets) in enumerate(test_loader):
        with torch.no_grad():
            inputVariableFrame = inputFrame.permute(1, 0, 2, 3, 4).cuda()
            inputVariableFlow = inputFlow.cuda()
            output_label = model(inputVariableFlow, inputVariableFrame)
        _, predictedTwoStream = torch.max(output_label.data, 1)
        all_predicted.append(F.softmax(output_label.data, 1))
        numCorrTwoStream += (predictedTwoStream == targets.cuda()).sum()
        predicted_labels.append(predictedTwoStream.cpu())
        true_labels.append(targets)
    test_accuracyTwoStream = torch.true_divide(numCorrTwoStream, test_samples) * 100
    print('Accuracy {:.02f}%'.format(test_accuracyTwoStream))

    df = pd.DataFrame([], columns=getClassNames(dataset_dir))
    for clss in range(num_classes):
        srtd, indices = torch.sort(torch.mean(torch.cat(all_predicted)[clss == torch.cat(true_labels)], 0),
                                   descending=True)
        print(getClassNames(dataset_dir)[clss], ':',
              list(zip(getClassNames(dataset_dir)[indices.cpu().numpy()], srtd.cpu().numpy()))[:5])
        df = pd.concat((df, pd.DataFrame(
            [torch.mean(torch.cat(all_predicted)[clss == torch.cat(true_labels)], 0).cpu().numpy()],
            columns=getClassNames(dataset_dir), index=[getClassNames(dataset_dir)[clss]])), 0)
    fig, ax = plt.subplots(figsize=(20, 20))
    ax = sns.heatmap(df, vmin=0, linewidth=0.1, square=True)
    plt.savefig(dataset + 'scores_hm.png', bbox_inches='tight')
    plt.show()

    verbs = set()
    objs = set()
    for clss in getClassNames(dataset_dir):
        verbs.add(clss.split('_')[0])
        for obj in clss.split('_')[1].split(','):
            objs.add(obj)
    verbs = sorted(verbs)
    objs = sorted(objs)

    df_verb = pd.DataFrame([], columns=['v1', 'v2', 'mean'])
    for v1 in verbs:
        v_means = torch.mean(
            torch.cat(all_predicted)[[v1 in s for s in getClassNames(dataset_dir)[torch.cat(true_labels)]]], 0)
        for v2 in verbs:
            df_verb = pd.concat((df_verb, pd.DataFrame(
                [[v1, v2, torch.sum(v_means[[v2 in s for s in getClassNames(dataset_dir)]])]],
                columns=['v1', 'v2', 'mean'])))
    df_verb = df_verb.pivot(index='v1', columns='v2', values='mean')
    fig, ax = plt.subplots(figsize=(7, 7))
    ax = sns.heatmap(df_verb.astype(float), vmin=0, annot=True, fmt='.2f', linewidth=0.1, square=True)
    plt.xlabel('')
    plt.ylabel('')
    plt.savefig(dataset + 'verbs_scores_hm.png', bbox_inches='tight')
    plt.show()

    df_obj = pd.DataFrame([], columns=['o1', 'o2', 'mean'])
    for o1 in objs:
        o_means = torch.mean(
            torch.cat(all_predicted)[[o1 in s for s in getClassNames(dataset_dir)[torch.cat(true_labels)]]], 0)
        for o2 in objs:
            df_obj = pd.concat((df_obj, pd.DataFrame(
                [[o1, o2, torch.sum(o_means[[o2 in s for s in getClassNames(dataset_dir)]])]],
                columns=['o1', 'o2', 'mean'])))
    df_obj = df_obj.pivot(index='o1', columns='o2', values='mean')
    fig, ax = plt.subplots(figsize=(10, 10))
    ax = sns.heatmap(df_obj.astype(float), vmin=0, annot=True, fmt='.2f', linewidth=0.1, square=True)
    plt.xlabel('')
    plt.ylabel('')
    plt.savefig(dataset + 'objs_scores_hm.png', bbox_inches='tight')
    plt.show()

    cnf_matrix = confusion_matrix(true_labels, predicted_labels).astype(float)
    cnf_matrix_normalized = cnf_matrix / cnf_matrix.sum(axis=1)[:, np.newaxis]

    plotAccuracyPerClassHist(cnf_matrix_normalized.diagonal(), dataset, dataset_dir)
    plotConfMatr(cnf_matrix_normalized, dataset, dataset_dir)
    # ticks = np.linspace(0, 60, num=61)
    # plt.imshow(cnf_matrix_normalized, interpolation='none', cmap='binary')
    # plt.colorbar()
    # plt.xticks(ticks, fontsize=6)
    # plt.yticks(ticks, fontsize=6)
    # plt.grid(True)
    # plt.clim(0, 1)
    # plt.savefig(dataset + '-twoStreamJoint.png', bbox_inches='tight')
    # plt.show()


def __main__():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='gtea61', help='Dataset')
    parser.add_argument('--datasetDir', type=str, default='./dataset/gtea_warped_flow_61/split2/test',
                        help='Dataset directory')
    parser.add_argument('--modelStateDict', type=str,
                        default='./models/gtea61/best_model_state_dict_twoStream_split2.pth',
                        help='Model path')
    parser.add_argument('--seqLen', type=int, default=25, help='Length of sequence')
    parser.add_argument('--stackSize', type=int, default=5, help='Number of optical flow images in input')
    parser.add_argument('--memSize', type=int, default=512, help='ConvLSTM hidden state size')
    parser.add_argument('--variant', type=str, default=None,
                        help="Attention mechanism on the candidate memory (c) vs. on the cell state (d)")

    args = parser.parse_args()

    dataset = args.dataset
    model_state_dict = args.modelStateDict
    dataset_dir = args.datasetDir
    seqLen = args.seqLen
    stackSize = args.stackSize
    memSize = args.memSize
    variant = args.variant

    main_run(dataset, model_state_dict, dataset_dir, stackSize, seqLen, memSize, variant)


__main__()

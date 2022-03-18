import torch.nn as nn
from os.path import join, dirname, isfile
import torch
import torch.nn.functional as F
import numpy as np
from torchvision.models.resnet import resnet18, resnet34
from utils.quantization import *
import tqdm
import time

class Classifier(nn.Module):
    def __init__(self,
                 voxel_dimension=(9,180,240),  # dimension of voxel will be C x 2 x H x W
                 in_channel=1, # input channel
                 num_classes=101,
                 representation='EST', #'EventCount', 'VoxGrid', 'EventFrame'
                 res_model = 'resnet34',
                 pretrained=True):

        nn.Module.__init__(self)
        if 'EST' in representation:
            self.quantization_layer = QuantizationLayerEST(voxel_dimension)
        elif 'EventCount' in representation:
            self.quantization_layer = QuantizationLayerEventCount(voxel_dimension)
        elif 'VoxGrid' in representation:
            self.quantization_layer = QuantizationLayerVoxGrid(voxel_dimension)
        else: # EventFrame
            self.quantization_layer = QuantizationLayerEventFrame(voxel_dimension)

        if 'resnet34' in res_model:
            self.classifier = resnet34(pretrained=pretrained)
        else:
            self.classifier = resnet18(pretained=pretrained)

        # replace fc layer and first convolutional layer
        input_channels = in_channel # for event count and event frame
        if len(voxel_dimension) == 3: # for vox grid and EST
            input_channels = in_channel * voxel_dimension[0]

        self.classifier.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.classifier.fc = nn.Linear(self.classifier.fc.in_features, num_classes)

    def forward(self, x):
        vox = self.quantization_layer.forward(x)
        pred = self.classifier.forward(vox)
        return pred



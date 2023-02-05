import numpy as np
import torch
import random
import math
from scipy import ndimage


def DeviceInitialization(GPUNum):
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        device = torch.device(GPUNum)
    else:
        device = torch.device('cpu')
    random.seed(2021)
    np.random.seed(2021)
    torch.manual_seed(2021)
    return device


def DataReading(path, set, fracTrain, fracTest):
    dataType = np.genfromtxt(path + set + '.txt', dtype=str)
    IDs = []
    labels = []
    for ii in range(len(dataType)):
        type = dataType[ii]
        data = np.genfromtxt(path + 'Csvs\\' + type + '.txt', dtype=str)
        for jj in range(len(data)):
            tempID = type + '\\' + data[jj]
            tempLabel = ii
            IDs.append(tempID)
            labels.append(tempLabel)

    shuffleIdx = np.arange(len(IDs))
    shuffleRng = np.random.RandomState(2022)
    shuffleRng.shuffle(shuffleIdx)
    IDs = np.array(IDs)[shuffleIdx]
    labels = np.array(labels)[shuffleIdx]

    trainNum = math.ceil(fracTrain * len(IDs) / 100)
    testNum = math.ceil(fracTest * len(IDs) / 100)
    testNum = len(IDs) - testNum
    trainIdxIDs = IDs[:trainNum]
    trainIdxLabels = labels[:trainNum]
    testIdxIDs = IDs[testNum:]
    testIdxLabels = labels[testNum:]

    return trainIdxIDs, trainIdxLabels, testIdxIDs, testIdxLabels

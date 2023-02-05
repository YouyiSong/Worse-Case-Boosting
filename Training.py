import torch
import numpy
from Tools import DeviceInitialization, DataReading
from Dataset import DataSet
from torch.utils.data import DataLoader
from Model import ViT
from Loss import Loss
import time
import math
from sklearn.linear_model import LinearRegression


epoch_num = 120
batch_size = 32
learning_rate = 3e-4
path = 'D:\\WorstCaseBoosting\\Data\\'
fracTrain = 80
fracTest = 20
set = 'SipakMed' ## 'SipakMed' or 'LCPSI'
class_num = 5 ## 5 or 4
netName = '_ViT_'
Net = ViT(image_size=64, patch_size=8, num_classes=class_num, dim=256, depth=10, heads=32, mlp_dim=512, dropout=0.1, emb_dropout=0.1)
lossType = 'CE' ## 'CE' or 'Focal'
device = DeviceInitialization('cuda:0')
trainIdxIDs, trainIdxLabels, testIdxIDs, testIdxLabels = DataReading(path=path, set=set, fracTrain=fracTrain, fracTest=fracTest)
trainSet = DataSet(set=set, path=path, IDs=trainIdxIDs, labels=trainIdxLabels, width=64, height=64, num_label=class_num, phase='Testing')
TrainSet = torch.utils.data.DataLoader(dataset=trainSet, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
testSet = DataSet(set=set, path=path, IDs=testIdxIDs, labels=testIdxLabels, width=64, height=64, num_label=class_num, phase='Testing')
TestSet = torch.utils.data.DataLoader(dataset=testSet, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
gradVec = torch.ones(len(trainIdxIDs)) * 1e20
dataIdx = numpy.arange(0, len(trainIdxIDs))
approProb = torch.ones(len(trainIdxIDs)) / len(trainIdxIDs)
Net.to(device)
optim = torch.optim.Adam(Net.parameters(), lr=learning_rate)
criterion = Loss(mode=lossType)
start_time = time.time()
IterNum = 0
sampleSize = 10
approSize = 5
for epoch in range(epoch_num):
    Net.train()
    tempLoss = 0
    for iteration in range(math.ceil(len(trainIdxIDs) / batch_size / sampleSize)):
        gradProb = gradVec / gradVec.sum()
        batchIdx = numpy.random.choice(dataIdx, size=sampleSize*batch_size, p=gradProb.numpy())
        batchIdxIDs = trainIdxIDs[batchIdx]
        batchIdxLabels = trainIdxLabels[batchIdx]
        gradInfo = gradVec[batchIdx]
        batchSet = DataSet(set=set, path=path, IDs=batchIdxIDs, labels=batchIdxLabels, width=64, height=64,
                           num_label=class_num, phase='Training')
        batchData = torch.utils.data.DataLoader(dataset=batchSet, batch_size=batch_size, shuffle=False,
                                                num_workers=0, pin_memory=True)

        for idx, (images, targets) in enumerate(batchData, 0):
            images = images.to(device)
            targets = targets.to(device)
            outputs = Net(images)
            loss = criterion(outputs, targets)
            weight = gradInfo[idx*batch_size : (idx+1)*batch_size]
            weight = batch_size * weight / weight.sum()
            weight = torch.clamp(weight, min=0.8, max=1.2)
            weight = batch_size * weight / weight.sum()
            loss = (loss * weight.to(device)).mean()
            loss = loss.mean()
            optim.zero_grad()
            loss.backward()
            optim.step()
            tempLoss += loss
        IterNum += (idx + 1)
        approIdx = numpy.random.choice(dataIdx, size=approSize*batch_size, p=approProb.numpy())
        approIdxIDs = trainIdxIDs[approIdx]
        approIdxLabels = trainIdxLabels[approIdx]
        trainGrad = gradVec[approIdx]
        tarGrad = torch.zeros_like(trainGrad)
        approSet = DataSet(set=set, path=path, IDs=approIdxIDs, labels=approIdxLabels, width=64, height=64,
                           num_label=class_num, phase='Testing')
        approData = torch.utils.data.DataLoader(dataset=approSet, batch_size=batch_size, shuffle=False,
                                                num_workers=0, pin_memory=True)

        for idx, (images, targets) in enumerate(approData, 0):
            images = images.to(device)
            targets = targets.to(device)
            outputs = Net(images)
            loss = criterion(outputs, targets)
            LastLayer = Net.get_last_layer()
            for ii in range(images.size(0)):
                tempGrad = torch.autograd.grad(loss[ii], LastLayer.parameters(), retain_graph=True)
                tarGrad[idx*batch_size + ii] = torch.norm(tempGrad[0].detach()) + torch.norm(tempGrad[1].detach())
        
        trainGrad = torch.stack((trainGrad, torch.ones_like(trainGrad)*trainGrad.mean()), dim=1)
        trainGrad = trainGrad.transpose(dim0=0, dim1=0)
        GradRegressor = LinearRegression().fit(trainGrad, tarGrad)
        preGrad = torch.stack((gradVec, torch.ones_like(gradVec)*gradVec.mean()), dim=1)
        preGrad = preGrad.transpose(dim0=0, dim1=0)
        gradVec = torch.from_numpy(GradRegressor.predict(preGrad))
        gradVec[approIdx] = tarGrad
        gradVec[gradVec<=0] = 1e-3
